//! Streaming archive importer: no media, inference, network access or secrets in output.
use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use chrono_tz::Tz;
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, env, error::Error, fs, io::{BufRead, BufReader, Read}, path::Path, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn string<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap_or("")
}

fn timestamp(raw: &str, zone: Tz) -> Option<String> {
    let dt = DateTime::parse_from_rfc3339(raw).ok().map(|v| v.with_timezone(&Utc)).or_else(|| {
        NaiveDateTime::parse_from_str(raw, "%Y-%m-%dT%H:%M:%S%.f").ok()
            .and_then(|v| zone.from_local_datetime(&v).single()).map(|v| v.with_timezone(&Utc))
    })?;
    Some(dt.format("%Y-%m-%dT%H:%M:%S+00:00").to_string())
}

fn digest_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 { break; }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn import(db: &mut Connection, path: &Path, zone: Tz) -> Result<(usize, usize)> {
    let path_string = path.to_string_lossy();
    let signature = digest_file(path)?;
    let previous: Option<String> = db.query_row("SELECT signature FROM imports WHERE path=?", [&*path_string], |r|r.get(0)).optional()?;
    if previous.as_deref() == Some(&signature) { return Ok((0, 0)); }
    let tx = db.transaction()?;
    let (mut count, mut malformed) = (0, 0);
    {
        let mut existing = tx.prepare_cached("SELECT note FROM events WHERE id=?")?;
        let mut insert = tx.prepare_cached("INSERT OR REPLACE INTO events(id,time,camera,bird,confidence,note,photo,source,activity,bbox) VALUES(?,?,?,?,?,?,?,'archive',?,?)")?;
        let mut delete_search = tx.prepare_cached("DELETE FROM search WHERE id=?")?;
        let mut insert_search = tx.prepare_cached("INSERT INTO search(id,note,bird) VALUES(?,?,?)")?;
        for line in BufReader::new(fs::File::open(path)?).lines() {
            let line = line?;
            let Ok(entry) = serde_json::from_str::<Value>(&line) else { malformed += 1; continue; };
            let Some(when) = timestamp(string(&entry, "time"), zone) else { malformed += 1; continue; };
            let fallback = vec![json!({"birds":entry.get("birds").unwrap_or(&Value::Null),"note":entry.get("note").unwrap_or(&Value::Null),
                "photo":entry.get("photos").and_then(Value::as_array).and_then(|a|a.first()).unwrap_or(&Value::Null),"camera":"Archive"})];
            let observations = entry.get("observations").and_then(Value::as_array).filter(|a|!a.is_empty()).unwrap_or(&fallback);
            for (index, obs) in observations.iter().enumerate() {
                let labels: BTreeSet<String> = obs.get("birds").and_then(Value::as_array).into_iter().flatten()
                    .filter_map(Value::as_str).map(|s|s.trim().to_lowercase()).collect();
                for bird in labels {
                    let photo = string(obs,"photo");
                    let key = format!("{:x}",Sha256::digest(format!("{when}:{index}:{bird}:{photo}").as_bytes()))[..24].to_string();
                    let note: String = string(obs,"note").chars().take(4000).collect();
                    let old: Option<String> = existing.query_row([&key], |r|r.get(0)).optional()?;
                    if old.as_deref() == Some(&note) { continue; }
                    let empty = Value::Null;
                    let detection = obs.get("detections").and_then(Value::as_array).into_iter().flatten()
                        .find(|d|string(d,"label").to_lowercase()==bird).unwrap_or(&empty);
                    let bbox = serde_json::to_string(detection.get("bbox").unwrap_or(&json!([])))?;
                    let camera: String = string(obs,"camera").chars().take(100).collect();
                    let activity: String = string(detection,"activity").chars().take(80).collect();
                    let label: String = bird.chars().take(60).collect();
                    insert.execute(params![key,when,camera,label,detection.get("confidence").and_then(Value::as_f64),note,photo,activity,bbox])?;
                    if old.is_some() { delete_search.execute([&key])?; }
                    insert_search.execute(params![key,note,label])?;
                    count += 1;
                }
            }
        }
    }
    tx.execute("INSERT OR REPLACE INTO imports VALUES(?,?)", params![path_string,signature])?;
    tx.commit()?;
    Ok((count,malformed))
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 { return Err("Usage: aviary-archive CONFIG_JSON JOURNAL_DIRECTORY".into()); }
    let config: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let zone: Tz = string(&config,"timezone").parse()?;
    let root = Path::new(string(&config,"runtime"));
    fs::create_dir_all(root)?;
    let mut db = Connection::open(root.join("aviary.sqlite"))?;
    db.busy_timeout(std::time::Duration::from_secs(15))?;
    db.execute_batch("PRAGMA journal_mode=WAL;
      CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,time TEXT NOT NULL,camera TEXT NOT NULL,bird TEXT NOT NULL,confidence REAL,note TEXT NOT NULL DEFAULT '',photo TEXT NOT NULL DEFAULT '',source TEXT NOT NULL,activity TEXT NOT NULL DEFAULT '',bbox TEXT NOT NULL DEFAULT '[]');
      CREATE INDEX IF NOT EXISTS events_time ON events(time DESC,id DESC);
      CREATE INDEX IF NOT EXISTS events_bird_time ON events(bird,time DESC,id DESC);
      CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(id UNINDEXED,note,bird);
      CREATE TABLE IF NOT EXISTS imports(path TEXT PRIMARY KEY,signature TEXT NOT NULL);")?;
    let mut paths: Vec<_> = fs::read_dir(&args[2])?.filter_map(|e|e.ok().map(|e|e.path()))
        .filter(|p|p.extension().is_some_and(|e|e=="jsonl")).collect();
    paths.sort();
    let start = Instant::now();
    let (mut total,mut malformed)=(0,0);
    for path in &paths { let (n,bad)=import(&mut db,path,zone)?;total+=n;malformed+=bad; }
    println!("{}",json!({"imported":total,"malformed":malformed,"files":paths.len(),"elapsed_seconds":start.elapsed().as_secs_f64()}));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn naive_timestamps_use_configured_zone() {
        assert_eq!(timestamp("2026-01-02T01:02:03.1234",chrono_tz::Asia::Tokyo).unwrap(),"2026-01-01T16:02:03+00:00");
    }
    #[test]
    fn explicit_offsets_are_preserved() {
        assert_eq!(timestamp("2026-01-02T01:02:03+02:00",chrono_tz::UTC).unwrap(),"2026-01-01T23:02:03+00:00");
        assert!(timestamp("not a date",chrono_tz::UTC).is_none());
    }
}
