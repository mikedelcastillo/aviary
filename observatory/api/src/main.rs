//! Native HTTP, authentication, archive queries and media serving. Python
//! adapters remain loopback-only for camera protocols and local model work.
use axum::extract::ws::{Message as ClientMessage, WebSocketUpgrade};
use axum::{
    Json, Router,
    body::{Body, to_bytes},
    extract::{ConnectInfo, DefaultBodyLimit, Path, Query, Request, State},
    http::{HeaderMap, HeaderValue, Method, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
};
use base64::{
    Engine,
    engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
};
use chrono::{Duration as ChronoDuration, NaiveDate, TimeZone, Utc};
use futures_util::{SinkExt, StreamExt};
use regex::Regex;
use rusqlite::{Connection, params, params_from_iter, types::ValueRef};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use subtle::ConstantTimeEq;
use tokio_tungstenite::tungstenite::Message as RelayMessage;
use tower::ServiceExt;
use tower_http::services::{ServeDir, ServeFile};

type App = Arc<Context>;
type Failure = (StatusCode, Json<Value>);
type Result<T> = std::result::Result<T, Failure>;
struct Context {
    config: Value,
    runtime: PathBuf,
    web: PathBuf,
    csp: String,
    version: String,
    client: reqwest::Client,
    adapter: String,
    attempts: Mutex<HashMap<String, Vec<f64>>>,
    connections: Mutex<Vec<Connection>>,
}
#[derive(Clone)]
struct Session(Option<String>);
fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}
fn fail(code: StatusCode, message: &str) -> Failure {
    (code, Json(json!({"detail":message})))
}
fn internal<T>(_: T) -> Failure {
    fail(
        StatusCode::INTERNAL_SERVER_ERROR,
        "The request could not be completed",
    )
}
fn digest(value: &str) -> String {
    hex::encode(Sha256::digest(value.as_bytes()))
}
fn header<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|h| h.to_str().ok())
}
fn cookie(headers: &HeaderMap) -> Option<String> {
    header(headers, "cookie")?
        .split(';')
        .find_map(|s| s.trim().strip_prefix("aviary_session=").map(str::to_owned))
}
fn origin_allowed(headers: &HeaderMap) -> bool {
    match header(headers, "origin") {
        None => true,
        Some(origin) => {
            origin
                .parse::<axum::http::Uri>()
                .ok()
                .and_then(|u| u.authority().map(|a| a.as_str().to_owned()))
                .as_deref()
                == header(headers, "host")
        }
    }
}
fn connect(path: &PathBuf) -> rusqlite::Result<Connection> {
    let db = Connection::open(path.join("aviary.sqlite"))?;
    db.busy_timeout(Duration::from_secs(15))?;
    // The index is small per query; bound caches across the eight worker slots.
    db.execute_batch("PRAGMA cache_size=-256;")?;
    Ok(db)
}
async fn db<T: Send + 'static>(
    state: &App,
    work: impl FnOnce(&Connection) -> Result<T> + Send + 'static,
) -> Result<T> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let cached = state.connections.lock().map_err(internal)?.pop();
        let connection = match cached {
            Some(connection) => connection,
            None => connect(&state.runtime).map_err(internal)?,
        };
        let result = work(&connection);
        // SQL work has finished before reacquiring this short pool lock.
        let mut pool = state.connections.lock().map_err(internal)?;
        if pool.len() < 8 {
            pool.push(connection);
        }
        result
    })
    .await
    .map_err(internal)?
}
fn rows(db: &Connection, sql: &str, args: &[String]) -> Result<Vec<Value>> {
    let mut statement = db.prepare(sql).map_err(internal)?;
    let names: Vec<String> = statement.column_names().map(|s| s.to_string()).collect();
    let mapped = statement
        .query_map(params_from_iter(args), |row| {
            let mut output = serde_json::Map::new();
            for (i, name) in names.iter().enumerate() {
                let value = match row.get_ref(i)? {
                    ValueRef::Null => Value::Null,
                    ValueRef::Integer(v) => json!(v),
                    ValueRef::Real(v) => json!(v),
                    ValueRef::Text(v) => json!(String::from_utf8_lossy(v)),
                    ValueRef::Blob(_) => Value::Null,
                };
                output.insert(name.clone(), value);
            }
            Ok(Value::Object(output))
        })
        .map_err(internal)?;
    mapped
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(internal)
}
fn decode_field(value: &mut Value, field: &str, fallback: Value) {
    value[field] = value[field]
        .as_str()
        .and_then(|v| serde_json::from_str(v).ok())
        .unwrap_or(fallback);
}
fn event(mut value: Value) -> Value {
    value["image"] = if value["photo"].as_str().is_some_and(|s| !s.is_empty()) {
        json!(format!(
            "/media/{}",
            value["id"].as_str().unwrap_or_default()
        ))
    } else {
        Value::Null
    };
    value.as_object_mut().unwrap().remove("photo");
    decode_field(&mut value, "bbox", json!([]));
    value
}
fn camera_exists(state: &Context, id: &str) -> bool {
    state.config["cameras"]
        .as_array()
        .is_some_and(|cs| cs.iter().any(|c| c["id"] == id))
}

async fn protect(State(state): State<App>, mut req: Request, next: Next) -> Response {
    if !matches!(*req.method(), Method::GET | Method::HEAD | Method::OPTIONS)
        && !origin_allowed(req.headers())
    {
        return fail(StatusCode::FORBIDDEN, "Origin rejected").into_response();
    }
    let path = req.uri().path().to_owned();
    let private = (path.starts_with("/api/") || path.starts_with("/media/"))
        && !matches!(
            path.as_str(),
            "/api/login" | "/api/session" | "/api/health" | "/api/discovery"
        );
    let session = if let Some(token) = cookie(req.headers()) {
        let key = digest(&token);
        let check = key.clone();
        match db(&state, move |db| {
            Ok(db
                .query_row("SELECT expires FROM sessions WHERE token=?", [check], |r| {
                    r.get::<_, f64>(0)
                })
                .ok()
                .is_some_and(|t| t > now()))
        })
        .await
        {
            Ok(true) => Some(key),
            _ => None,
        }
    } else {
        None
    };
    if private && session.is_none() {
        return fail(StatusCode::UNAUTHORIZED, "Please unlock Aviary").into_response();
    }
    req.extensions_mut().insert(Session(session));
    let mut response = next.run(req).await;
    let headers = response.headers_mut();
    headers.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    headers.insert("referrer-policy", HeaderValue::from_static("no-referrer"));
    headers.insert("x-frame-options", HeaderValue::from_static("DENY"));
    headers.insert(
        "content-security-policy",
        HeaderValue::from_str(&state.csp).unwrap(),
    );
    if private || path.starts_with("/api/") {
        headers.insert("cache-control", HeaderValue::from_static("no-store"));
    }
    response
}
#[derive(Deserialize)]
struct Login {
    pin: String,
}
async fn login(
    State(state): State<App>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(body): Json<Login>,
) -> Result<Response> {
    if body.pin.len() > 128 {
        return Err(fail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Passcode is too long",
        ));
    }
    let address = peer.ip().to_string();
    {
        let mut all = state.attempts.lock().map_err(internal)?;
        all.retain(|_, v| v.last().is_some_and(|t| now() - t < 300.0));
        let attempts = all.entry(address.clone()).or_default();
        attempts.retain(|t| now() - t < 300.0);
        if attempts.len() >= 8 {
            return Err(fail(
                StatusCode::TOO_MANY_REQUESTS,
                "Please wait five minutes before trying again.",
            ));
        }
        attempts.push(now());
    }
    let salt =
        hex::decode(state.config["pin_salt"].as_str().unwrap_or_default()).map_err(internal)?;
    let expected =
        hex::decode(state.config["pin_hash"].as_str().unwrap_or_default()).map_err(internal)?;
    let valid = tokio::task::spawn_blocking(move || {
        let mut key = [0u8; 32];
        pbkdf2::pbkdf2_hmac::<pbkdf2::sha2::Sha256>(body.pin.as_bytes(), &salt, 600_000, &mut key);
        bool::from(key.as_slice().ct_eq(&expected))
    })
    .await
    .map_err(internal)?;
    if !valid {
        return Err(fail(
            StatusCode::UNAUTHORIZED,
            "That passcode did not match.",
        ));
    }
    state.attempts.lock().map_err(internal)?.remove(&address);
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).map_err(internal)?;
    let token = URL_SAFE_NO_PAD.encode(bytes);
    let key = digest(&token);
    db(&state, move |db| {
        db.execute("DELETE FROM sessions WHERE expires<?", [now()])
            .map_err(internal)?;
        db.execute(
            "INSERT INTO sessions VALUES(?,?)",
            params![key, now() + 90.0 * 86400.0],
        )
        .map_err(internal)?;
        Ok(())
    })
    .await?;
    let secure = peer.ip().is_loopback() && header(&headers, "x-forwarded-proto") == Some("https");
    let mut response = Json(json!({"ok":true})).into_response();
    response.headers_mut().insert(
        "set-cookie",
        HeaderValue::from_str(&format!(
            "aviary_session={token}; Path=/; Max-Age={}; HttpOnly; SameSite=Strict{}",
            90 * 86400,
            if secure { "; Secure" } else { "" }
        ))
        .unwrap(),
    );
    Ok(response)
}
async fn current_session(req: Request) -> Json<Value> {
    Json(json!({"authenticated":req.extensions().get::<Session>().is_some_and(|s|s.0.is_some())}))
}
async fn logout(State(state): State<App>, req: Request) -> Result<Response> {
    let key = req
        .extensions()
        .get::<Session>()
        .and_then(|s| s.0.clone())
        .unwrap_or_default();
    db(&state, move |db| {
        db.execute("DELETE FROM sessions WHERE token=?", [key])
            .map_err(internal)?;
        Ok(())
    })
    .await?;
    let mut response = Json(json!({"ok":true})).into_response();
    response.headers_mut().insert(
        "set-cookie",
        HeaderValue::from_static("aviary_session=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict"),
    );
    Ok(response)
}
#[derive(Deserialize, Default)]
struct Events {
    #[serde(default)]
    bird: String,
    #[serde(default)]
    date: String,
    #[serde(default)]
    before: String,
    #[serde(default)]
    search: String,
    #[serde(default)]
    favorites: bool,
    limit: Option<usize>,
}
async fn events(State(state): State<App>, Query(query): Query<Events>) -> Result<Json<Value>> {
    let mut conditions = Vec::new();
    let mut args = Vec::new();
    if !query.bird.is_empty() {
        conditions.push("e.bird=?");
        args.push(query.bird);
    }
    if !query.date.is_empty() {
        let day = NaiveDate::parse_from_str(&query.date, "%Y-%m-%d")
            .map_err(|_| fail(StatusCode::UNPROCESSABLE_ENTITY, "Invalid date"))?;
        let zone: chrono_tz::Tz = state.config["timezone"]
            .as_str()
            .unwrap_or("UTC")
            .parse()
            .map_err(internal)?;
        let start = zone
            .from_local_datetime(&day.and_hms_opt(0, 0, 0).unwrap())
            .earliest()
            .ok_or_else(|| fail(StatusCode::UNPROCESSABLE_ENTITY, "Invalid date"))?;
        let end = zone
            .from_local_datetime(
                &(day + ChronoDuration::days(1))
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
            )
            .earliest()
            .ok_or_else(|| fail(StatusCode::UNPROCESSABLE_ENTITY, "Invalid date"))?;
        conditions.push("e.time>=? AND e.time<?");
        args.extend([
            start.with_timezone(&Utc).to_rfc3339(),
            end.with_timezone(&Utc).to_rfc3339(),
        ]);
    }
    if !query.before.is_empty() {
        let (time, id) = query
            .before
            .split_once('|')
            .ok_or_else(|| fail(StatusCode::UNPROCESSABLE_ENTITY, "Invalid cursor"))?;
        conditions.push("(e.time,e.id)<(?,?)");
        args.extend([time.to_owned(), id.to_owned()]);
    }
    let terms: Vec<_> = query
        .search
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .take(12)
        .map(|s| format!("\"{s}\""))
        .collect();
    if !terms.is_empty() {
        conditions.push("e.id IN(SELECT id FROM search WHERE search MATCH ?)");
        args.push(terms.join(" AND "));
    }
    if query.favorites {
        conditions.push("e.id IN(SELECT event FROM favorites)");
    }
    let filter = if conditions.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", conditions.join(" AND "))
    };
    args.push(query.limit.unwrap_or(24).clamp(1, 60).to_string());
    db(&state,move|db|{
        let values=rows(db,&format!("SELECT e.*,EXISTS(SELECT 1 FROM favorites f WHERE f.event=e.id) favorite FROM events e{filter} ORDER BY time DESC,id DESC LIMIT ?"),&args)?;
        let next=values.last().map(|r|format!("{}|{}",r["time"].as_str().unwrap(),r["id"].as_str().unwrap()));
        let values:Vec<_>=values.into_iter().map(|v|{let mut v=event(v);v["favorite"]=json!(v["favorite"].as_i64()==Some(1));v}).collect();Ok(Json(json!({"events":values,"next":next})))
    }).await
}
async fn favorite_event(State(state): State<App>, Path(id): Path<String>) -> Result<Json<Value>> {
    db(&state, move |db| {
        if !db
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM events WHERE id=?)",
                [&id],
                |r| r.get::<_, bool>(0),
            )
            .map_err(internal)?
        {
            return Err(fail(StatusCode::NOT_FOUND, "Observation not found"));
        }
        let exists = db
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM favorites WHERE event=?)",
                [&id],
                |r| r.get::<_, bool>(0),
            )
            .map_err(internal)?;
        db.execute(
            if exists {
                "DELETE FROM favorites WHERE event=?"
            } else {
                "INSERT INTO favorites VALUES(?)"
            },
            [id],
        )
        .map_err(internal)?;
        Ok(Json(json!({"favorite":!exists})))
    })
    .await
}
async fn zones(State(state): State<App>) -> Result<Json<Value>> {
    db(&state, |db| {
        let mut values = rows(db, "SELECT * FROM zones WHERE enabled=1", &[])?;
        for v in &mut values {
            decode_field(v, "rect", json!([]));
        }
        Ok(Json(json!(values)))
    })
    .await
}
async fn disable_zone(State(state): State<App>, Path(id): Path<String>) -> Result<Json<Value>> {
    db(&state, move |db| {
        db.execute("UPDATE zones SET enabled=0 WHERE id=?", [id])
            .map_err(internal)?;
        Ok(Json(json!({"ok":true})))
    })
    .await
}

async fn serve_file(path: PathBuf, req: Request) -> Result<Response> {
    ServeFile::new(path)
        .oneshot(req)
        .await
        .map(IntoResponse::into_response)
        .map_err(internal)
}

async fn live_socket(
    State(state): State<App>,
    Path(camera): Path<String>,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Result<Response> {
    if !origin_allowed(&headers) {
        return Err(fail(StatusCode::FORBIDDEN, "Origin rejected"));
    }
    if !camera_exists(&state, &camera) {
        return Err(fail(StatusCode::NOT_FOUND, "Camera not found"));
    }
    let relay = state.config["relay_ws"]
        .as_str()
        .ok_or_else(|| fail(StatusCode::SERVICE_UNAVAILABLE, "Relay unavailable"))?;
    let url: reqwest::Url = relay.parse().map_err(internal)?;
    if url.scheme() != "ws" || !matches!(url.host_str(), Some("127.0.0.1") | Some("localhost")) {
        return Err(fail(StatusCode::SERVICE_UNAVAILABLE, "Relay must be local"));
    }
    let target = format!("{relay}/api/ws?src={camera}");
    Ok(upgrade
        .max_message_size(2048)
        .max_frame_size(2048)
        .on_upgrade(move |socket| async move {
            let connected = tokio::time::timeout(
                Duration::from_secs(5),
                tokio_tungstenite::connect_async(target),
            )
            .await;
            let Ok(Ok((upstream, _))) = connected else {
                return;
            };
            let (mut client_send, mut client_receive) = socket.split();
            let (mut relay_send, mut relay_receive) = upstream.split();
            let to_relay = async {
                while let Some(Ok(message)) = client_receive.next().await {
                    match message {
                        ClientMessage::Text(text) => {
                            let Ok(value) = serde_json::from_str::<Value>(&text) else {
                                break;
                            };
                            // Playback negotiation only: no source changes, microphone or admin messages.
                            if text.len() > 2048
                                || !matches!(value["type"].as_str(), Some("mse" | "mp4" | "hls"))
                            {
                                break;
                            }
                            if relay_send
                                .send(RelayMessage::Text(text.to_string().into()))
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                        ClientMessage::Ping(_) | ClientMessage::Pong(_) => {}
                        _ => break,
                    }
                }
            };
            let to_client = async {
                while let Some(Ok(message)) = relay_receive.next().await {
                    let message = match message {
                        RelayMessage::Text(text) => ClientMessage::Text(text.to_string().into()),
                        RelayMessage::Binary(bytes) if bytes.len() <= 4 * 1024 * 1024 => {
                            ClientMessage::Binary(bytes)
                        }
                        RelayMessage::Ping(_) | RelayMessage::Pong(_) => continue,
                        _ => break,
                    };
                    // A slow viewer cannot accumulate an unbounded stream queue.
                    if !matches!(
                        tokio::time::timeout(Duration::from_secs(10), client_send.send(message))
                            .await,
                        Ok(Ok(()))
                    ) {
                        break;
                    }
                }
            };
            tokio::select! { _ = to_relay => {}, _ = to_client => {} }
        })
        .into_response())
}
async fn hls(
    State(state): State<App>,
    Path((camera, file)): Path<(String, String)>,
    req: Request,
) -> Result<Response> {
    if !camera_exists(&state, &camera)
        || !(file == "live.m3u8"
            || (file.starts_with('s')
                && file.ends_with(".ts")
                && file[1..file.len() - 3].chars().all(|c| c.is_ascii_digit())))
    {
        return Err(fail(StatusCode::NOT_FOUND, "Segment not found"));
    }
    let path = state.runtime.join("hls").join(camera).join(file);
    if !path.is_file() {
        return Err(fail(
            StatusCode::SERVICE_UNAVAILABLE,
            "Waiting for camera segments",
        ));
    }
    serve_file(path, req).await
}
async fn clip_video(
    State(state): State<App>,
    Path(id): Path<String>,
    req: Request,
) -> Result<Response> {
    let relative = db(&state, move |db| {
        db.query_row("SELECT path FROM clips WHERE id=?", [id], |r| {
            r.get::<_, String>(0)
        })
        .map_err(|_| fail(StatusCode::NOT_FOUND, "Clip not found"))
    })
    .await?;
    let path = std::fs::canonicalize(state.runtime.join(relative))
        .map_err(|_| fail(StatusCode::GONE, "This rolling recording has expired"))?;
    if !path.starts_with(&state.runtime) {
        return Err(fail(StatusCode::NOT_FOUND, "Clip not found"));
    }
    serve_file(path, req).await
}
#[derive(Deserialize)]
struct Seen {
    watched_seconds: f64,
}
async fn seen(
    State(state): State<App>,
    Path(id): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Seen>,
) -> Result<Json<Value>> {
    if !body.watched_seconds.is_finite() || !(2.0..=3600.0).contains(&body.watched_seconds) {
        return Err(fail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Invalid watch duration",
        ));
    }
    let viewer = digest(&cookie(&headers).unwrap_or_default());
    db(&state,move|db|{
        let duration=db.query_row("SELECT duration FROM clips WHERE id=?",[&id],|r|r.get::<_,f64>(0)).map_err(|_|fail(StatusCode::NOT_FOUND,"Clip not found"))?;
        db.execute("INSERT INTO clip_views(clip,viewer,seen_at,watched_seconds) VALUES(?,?,?,?) ON CONFLICT(clip,viewer) DO UPDATE SET watched_seconds=max(watched_seconds,excluded.watched_seconds)",params![id,viewer,Utc::now().to_rfc3339(),body.watched_seconds.min(duration)]).map_err(internal)?;Ok(Json(json!({"ok":true})))
    }).await
}
async fn adapter(State(state): State<App>, req: Request) -> Result<Response> {
    let path = req
        .uri()
        .path_and_query()
        .map(|s| s.as_str())
        .unwrap_or("/")
        .to_owned();
    let method = req.method().clone();
    let headers = req.headers().clone();
    let bytes = to_bytes(req.into_body(), 2 * 1024 * 1024)
        .await
        .map_err(|_| fail(StatusCode::PAYLOAD_TOO_LARGE, "Request too large"))?;
    let mut upstream = state
        .client
        .request(method, format!("{}{path}", state.adapter))
        .body(bytes);
    for name in [
        "cookie",
        "origin",
        "host",
        "content-type",
        "range",
        "if-none-match",
        "if-modified-since",
    ] {
        if let Some(value) = headers.get(name) {
            upstream = upstream.header(name, value);
        }
    }
    let response = upstream.send().await.map_err(|_| {
        fail(
            StatusCode::SERVICE_UNAVAILABLE,
            "The local camera/model adapter is unavailable",
        )
    })?;
    let status = response.status();
    let response_headers = response.headers().clone();
    let mut output = Response::new(Body::from_stream(response.bytes_stream()));
    *output.status_mut() = status;
    for name in [
        "content-type",
        "content-length",
        "content-range",
        "accept-ranges",
        "etag",
        "last-modified",
    ] {
        if let Some(value) = response_headers.get(name) {
            output.headers_mut().insert(name, value.clone());
        }
    }
    Ok(output)
}
async fn fallback(State(state): State<App>, req: Request) -> Result<Response> {
    if req.uri().path().starts_with("/api/") || req.uri().path().starts_with("/media/") {
        return adapter(State(state), req).await;
    }
    ServeDir::new(&state.web)
        .oneshot(req)
        .await
        .map(IntoResponse::into_response)
        .map_err(internal)
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .max_blocking_threads(8)
        .enable_all()
        .build()?
        .block_on(run())
}
async fn run() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let config: Value = serde_json::from_slice(&std::fs::read(
        std::env::var("AVIARY_HOME_CONFIG").unwrap_or("runtime/config.json".into()),
    )?)?;
    let runtime = std::fs::canonicalize(config["runtime"].as_str().ok_or("Missing runtime")?)?;
    let web = std::fs::canonicalize(std::env::var("AVIARY_WEB_ROOT").unwrap_or("web/out".into()))?;
    let adapter_url = std::env::var("AVIARY_ADAPTER_URL").unwrap_or("http://127.0.0.1:8741".into());
    let url: reqwest::Url = adapter_url.parse()?;
    if url.host_str() != Some("127.0.0.1") && url.host_str() != Some("localhost") {
        return Err("Adapter must be loopback-only".into());
    }
    let identifier = Regex::new(r"^[A-Za-z0-9_-]+$")?;
    for camera in config["cameras"].as_array().ok_or("Missing cameras")? {
        if !identifier.is_match(camera["id"].as_str().unwrap_or_default()) {
            return Err("Invalid camera ID".into());
        }
    }
    connect(&runtime)?.execute_batch("CREATE TABLE IF NOT EXISTS clip_views(clip TEXT,viewer TEXT,seen_at TEXT,watched_seconds REAL,PRIMARY KEY(clip,viewer)); CREATE INDEX IF NOT EXISTS clip_views_clip ON clip_views(clip);")?;
    let scripts = Regex::new(r"(?s)<script\b([^>]*)>(.*?)</script>")?;
    let mut hashes = Vec::new();
    for file in std::fs::read_dir(&web)? {
        let file = file?.path();
        if file.extension().is_some_and(|v| v == "html") {
            for captures in scripts.captures_iter(&std::fs::read_to_string(file)?) {
                if !captures[1].contains("src=") && !captures[2].is_empty() {
                    hashes.push(format!(
                        "'sha256-{}'",
                        STANDARD.encode(Sha256::digest(captures[2].as_bytes()))
                    ));
                }
            }
        }
    }
    hashes.sort();
    hashes.dedup();
    let csp = format!(
        "default-src 'self'; script-src 'self' {}; worker-src 'self' blob:; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self' wss: ws:; font-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'self'",
        hashes.join(" ")
    );
    let version = digest(&std::fs::read_to_string(web.join("index.html"))?)[..12].to_owned();
    let state = Arc::new(Context {
        config,
        runtime,
        web,
        csp,
        version,
        client: reqwest::Client::builder()
            .timeout(Duration::from_secs(180))
            .redirect(reqwest::redirect::Policy::none())
            .build()?,
        adapter: adapter_url,
        attempts: Mutex::new(HashMap::new()),
        connections: Mutex::new(Vec::new()),
    });
    let app = Router::new()
        .route(
            "/api/discovery",
            get(|| async { Json(json!({"service":"aviary-observatory","api":1})) }),
        )
        .route(
            "/api/health",
            get(|| async { Json(json!({"ok":true,"engine":"rust"})) }),
        )
        .route(
            "/api/version",
            get(|State(s): State<App>| async move {
                Json(json!({"version":s.version,"api":"rust","web":"nextjs-typescript"}))
            }),
        )
        .route("/api/session", get(current_session))
        .route("/api/login", post(login))
        .route("/api/logout", post(logout))
        .route("/api/events", get(events))
        .route("/api/events/{id}/favorite", post(favorite_event))
        .route("/api/zones", get(zones).post(adapter))
        .route("/api/zones/{id}", delete(disable_zone))
        .route("/api/cameras/{camera}/hls/{file}", get(hls))
        .route("/api/cameras/{camera}/stream", get(live_socket))
        .route("/api/clips/{id}/video", get(clip_video))
        .route("/api/clips/{id}/seen", post(seen))
        .fallback(fallback)
        .layer(DefaultBodyLimit::max(2 * 1024 * 1024))
        .layer(middleware::from_fn_with_state(state.clone(), protect))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(
        std::env::var("AVIARY_LISTEN").unwrap_or("127.0.0.1:8740".into()),
    )
    .await?;
    println!("aviary-api ready; native auth, archive queries and media; local adapters enabled");
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}
