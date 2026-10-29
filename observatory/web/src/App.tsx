"use client";
import React, { useEffect, useRef, useState } from "react";
import type { LucideIcon } from "lucide-react";
import {errorMessage,type Observation,type BirdProfile,type Camera,type Overview,type Zone,type ChatMessage} from "./types";
import {
  Sun,
  Radio,
  Bird,
  Images,
  MessageCircle,
  Settings,
  ArrowUpRight,
  ArrowRight,
  ChevronRight,
  ChevronLeft,
  Heart,
  Play,
  Pause,
  X,
  Search,
  Plus,
  Check,
  ShieldCheck,
  ShieldAlert,
  Maximize2,
  RefreshCw,
  Send,
  MapPin,
  Droplets,
  Utensils,
  Eye,
  Lock,
  Download,
  LogOut,
  Activity,
  CalendarDays,
} from "lucide-react";
import "./style.css";
import "./field.css";
import LivePlayer from "./LivePlayer";
import PTZControls from "./PTZControls";
import VideoFeed from "./VideoFeed";
import NerdStats from "./NerdStats";
import UpdateNotice from "./UpdateNotice";
import DailyTag from "./DailyTag";
import LiveWall from "./LiveWall";
import ActivityFeed from "./ActivityFeed";
import './immersive.css';

async function api<T = {ok:boolean}>(path:string, options:RequestInit = {}):Promise<T> {
  const response = await fetch("/api" + path, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  if (!response.ok) {
    let data: {detail?:string} = {};
    try {
      data = await response.json();
    } catch {}
    if (response.status === 401 && path != "/login")
      window.dispatchEvent(new Event("locked"));
    throw new Error(data.detail || "Connection interrupted. Please try again.");
  }
  return response.json();
}
const ago = (t:string|null|undefined) => {
  if (!t) return "Not seen yet";
  const n = Math.max(0, (Date.now() - new Date(t).getTime()) / 1000);
  return n < 60
    ? "Just now"
    : n < 3600
      ? `${Math.floor(n / 60)}m ago`
      : n < 86400
        ? `${Math.floor(n / 3600)}h ago`
        : new Date(t).toLocaleDateString(undefined, {
            month: "short",
            day: "numeric",
          });
};
const stamp = (t:string) =>
  new Date(t).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
const title = (s:string) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : "");
const tabs: [string,string,LucideIcon][] = [
  ["live", "Live", Radio],
  ["clips", "Reels", Play],
  ["today", "Feed", Sun],
];
function IconButton({ label, children, ...props }: {label:string}&React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button className="icon-button" aria-label={label} title={label} {...props}>
      {children}
    </button>
  );
}
function Empty({ icon: Icon = Bird, title: heading, children }: {icon?:LucideIcon;title:string;children?:React.ReactNode}) {
  return (
    <div className="empty">
      <Icon size={28} />
      <h3>{heading}</h3>
      <p>{children}</p>
    </div>
  );
}
function Photo({ src, alt, ...props }: Omit<React.ImgHTMLAttributes<HTMLImageElement>,"src">&{src?:string|null}) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [src]);
  return src && !failed ? (
    <img src={src} alt={alt} onError={() => setFailed(true)} {...props} />
  ) : (
    <div className="photo-missing">
      <Bird size={28} />
      <span>Photo unavailable</span>
    </div>
  );
}
function Avatar({ bird, size = "" }: {bird?:BirdProfile;size?:string}) {
  return (
    <div className={"avatar " + size}>
      {bird?.portrait ? (
        <Photo src={bird.portrait} alt={bird.name} />
      ) : (
        <Bird />
      )}
    </div>
  );
}
function Modal({ title: heading, children, close, wide = false }: {title:string;children:React.ReactNode;close:()=>void;wide?:boolean}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const onKey = (e:KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    ref.current?.focus();
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, []);
  return (
    <div className="scrim" onClick={close}>
      <section
        ref={ref}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-label={heading}
        className={"modal " + (wide ? "wide" : "")}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-head">
          <h2>{heading}</h2>
          <IconButton label="Close" onClick={close}>
            <X />
          </IconButton>
        </div>
        {children}
      </section>
    </div>
  );
}

function Login({ done }: {done:()=>void}) {
  const [pin, setPin] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  async function submit(e:React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await api("/login", { method: "POST", body: JSON.stringify({ pin }) });
      done();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <main className="login">
      <div className="login-art">
        <div className="orbit one" />
        <div className="orbit two" />
        <img src="/icon.svg" alt="" />
        <span>
          The birds,
          <br />
          at home.
        </span>
        <p>
          Live cameras and recorded video.
          <br />
          Private. Local. Yours.
        </p>
      </div>
      <form className="login-form" onSubmit={submit}>
        <div className="eyebrow">AVIARY HOME</div>
        <h1>Aviary</h1>
        <p>Enter your home passcode.</p>
        <label>
          Home passcode
          <input
            type="password"
            inputMode="numeric"
            autoComplete="current-password"
            value={pin}
            onChange={(e) => setPin(e.target.value)}
            placeholder="Enter your passcode"
            required
            autoFocus
          />
        </label>
        {error && (
          <p role="alert" className="error">
            {error}
          </p>
        )}
        <button className="primary" disabled={busy}>
          {busy ? "Opening…" : "Open Aviary"}
          <ArrowRight size={18} />
        </button>
        <small>
          <Lock size={14} />
          Private to your home. Remembered on this device.
        </small>
      </form>
    </main>
  );
}

function LiveView({ camera, large = false, compact = false }: {camera?:Camera;large?:boolean;compact?:boolean}) {
  const [mode, setMode] = useState("original"),
    [image, setImage] = useState(""),
    [failed, setFailed] = useState(false),
    [playback, setPlayback] = useState("waiting");
  const videoRef = useRef<HTMLDivElement>(null),
    panelRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let active = true,
      previous = "";
    setImage("");
    setFailed(false);
    if (mode !== "screened" || !camera?.online || camera.privacy !== "clear")
      return;
    async function frame() {
      try {
        const r = await fetch(
          `/api/cameras/${camera!.id}/snapshot?t=${Date.now()}`,
          { cache: "no-store" },
        );
        if (!r.ok) throw Error();
        const blob = await r.blob();
        if (!active) return;
        const url = URL.createObjectURL(blob);
        if (previous) URL.revokeObjectURL(previous);
        previous = url;
        setImage(url);
        setFailed(false);
      } catch {
        if (active) setFailed(true);
      }
    }
    frame();
    const t = setInterval(() => {
      if (!document.hidden) frame();
    }, 1300);
    return () => {
      active = false;
      clearInterval(t);
      if (previous) URL.revokeObjectURL(previous);
    };
  }, [camera?.id, camera?.online, camera?.privacy, mode]);
  const available = camera?.online && camera?.privacy === "clear" && !failed;
  return (
    <div className={"live-view " + (large ? "large" : "") + (compact ? " compact-live" : "")} ref={panelRef}>
      <div className="live-top">
        <span className="camera-label">
          <span
            className={
              "dot " +
              ((mode === "original" ? playback === "playing" : available)
                ? "on"
                : "")
            }
          />
          {camera?.name || "Camera"}
        </span>
        <IconButton
          label="Full screen"
          onClick={() => {
            if (panelRef.current?.requestFullscreen)
              panelRef.current.requestFullscreen();
            else
              videoRef.current
                ?.querySelector("video")
                ?.webkitEnterFullscreen?.();
          }}
        >
          <Maximize2 size={17} />
        </IconButton>
      </div>
      <div className="live-picture">
        {mode === "original" ? (
          <>
            <div ref={videoRef} className="video-host"><LivePlayer camera={camera?.id || ""} onState={setPlayback}/></div>
            {false && playback !== "playing" && (
              <div className="video-wait">
                <Radio />
                <span>
                  {playback === "error"
                    ? "Reconnecting to the camera…"
                    : "Connecting to live video…"}
                </span>
              </div>
            )}
          </>
        ) : available && image ? (
          <img
            className="frame"
            src={image}
            alt={`${camera?.name} screened camera view`}
          />
        ) : (
          <div className="live-placeholder">
            {camera?.privacy === "person" ? (
              <ShieldCheck size={38} />
            ) : (
              <Radio size={38} />
            )}
            <h3>
              {camera?.privacy === "person"
                ? "A little privacy."
                : camera?.online
                  ? "Preparing your view…"
                  : "Waiting for the camera"}
            </h3>
            <p>
              {camera?.privacy === "person"
                ? "A person was detected. The screened view will return when the frame is clear."
                : "The view reconnects automatically. Your memories are still here."}
            </p>
          </div>
        )}
        {mode === "screened" && available && image && (
          <div className="live-caption">
            <span>Screened view</span>
            <span>Updated every moment</span>
          </div>
        )}
      </div>
      <div className="live-bottom">
        <div className="segmented">
          <button
            className={mode === "screened" ? "selected" : ""}
            onClick={() => setMode("screened")}
          >
            <ShieldCheck size={15} />
            Screened
          </button>
          <button
            className={mode === "original" ? "selected" : ""}
            onClick={() => setMode("original")}
          >
            <Play size={15} />
            Smooth Live
          </button>
        </div>
        <span className="view-note">
          {mode === "original"
            ? "Original video · people may appear"
            : "Person-screened snapshots"}
        </span>
      </div>
      {camera && !compact && <PTZControls camera={camera.id}/>}
      {camera?.zone_views && Object.keys(camera.zone_views).length > 0 && !Object.values(camera.zone_views).some(view=>view.valid) && <p className="feed-context" role="status">Zone counting paused. This view has not matched its saved reference.</p>}
    </div>
  );
}

function EventCard({
  event,
  birds,
  cameras,
  open,
  compact = false,
  onFavorite,
}: {event:Observation;birds:BirdProfile[];cameras:Camera[];open:(event:Observation)=>void;compact?:boolean;onFavorite:(event:Observation)=>void}) {
  const bird = birds.find((b) => b.id === event.bird),
    camera = cameras.find((c) => c.id === event.camera);
  return (
    <article className={"event-card " + (compact ? "compact" : "")}>
      <button
        className="event-photo"
        onClick={() => open(event)}
        aria-label={`Open ${bird?.name || title(event.bird)} sighting`}
      >
        <Photo
          src={event.image}
          alt={`${bird?.name || event.bird} sighting`}
          loading="lazy"
        />
        <span className="photo-time">
          {new Date(event.time).toLocaleTimeString(undefined, {
            hour: "numeric",
            minute: "2-digit",
          })}
        </span>
      </button>
      <div className="event-body">
        <div className="event-author">
          <Avatar bird={bird} size="tiny" />
          <div>
            <strong>{bird?.name || title(event.bird)}</strong>
            <small>
              {camera?.name || "Archive"} · {ago(event.time)}
            </small>
          </div>
          <IconButton
            label={event.favorite ? "Unsave memory" : "Save memory"}
            onClick={() => onFavorite(event)}
          >
            <Heart size={18} fill={event.favorite ? "currentColor" : "none"} />
          </IconButton>
        </div>
        <button className="event-title" onClick={() => open(event)}>
          {bird
            ? `${bird.name}, a moment in the aviary.`
            : "A bird spotted in the aviary."}
        </button>
        <p>
          {event.source === "live"
            ? "A fresh sighting, captured locally."
            : bird
              ? "A moment from the family archive."
              : "Individual identity was not established."}
        </p>
      </div>
    </article>
  );
}

function Replay({ event, events, birds, cameras, close, onFavorite }: {event:Observation;events:Observation[];birds:BirdProfile[];cameras:Camera[];close:()=>void;onFavorite:(event:Observation)=>void}) {
  const [index, setIndex] = useState(
      Math.max(
        0,
        events.findIndex((e) => e.id === event.id),
      ),
    ),
    [playing, setPlaying] = useState(false);
  const current = events[index] || event;
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(
      () => setIndex((i) => (i + 1) % events.length),
      2500,
    );
    return () => clearInterval(timer);
  }, [playing, events.length]);
  return (
    <Modal title="Recorded moment" close={close} wide>
      <div className="replay-photo">
        <Photo
          src={current.image}
          alt={`${current.bird} at ${stamp(current.time)}`}
        />
      </div>
      <div className="replay-controls">
        <IconButton
          label="Previous moment"
          onClick={() =>
            setIndex((i) => (i - 1 + events.length) % events.length)
          }
        >
          <ChevronLeft />
        </IconButton>
        <button className="secondary" onClick={() => setPlaying(!playing)}>
          {playing ? <Pause size={17} /> : <Play size={17} />}{" "}
          {playing ? "Pause" : "Play moments"}
        </button>
        <IconButton
          label="Next moment"
          onClick={() => setIndex((i) => (i + 1) % events.length)}
        >
          <ChevronRight />
        </IconButton>
        <span>
          {index + 1} / {events.length} photos
        </span>
      </div>
      <div className="replay-description">
        <div>
          <h2>
            {birds.find((b) => b.id === current.bird)?.name ||
              title(current.bird)}
          </h2>
          <p>
            {stamp(current.time)} ·{" "}
            {cameras.find((c) => c.id === current.camera)?.name || "Archive"}
          </p>
        </div>
        <IconButton label="Save memory" onClick={() => onFavorite(current)}>
          <Heart fill={current.favorite ? "currentColor" : "none"} />
        </IconButton>
      </div>
      <p className="muted">
        Photo replay · individual identity is a detector estimate
        {current.confidence
          ? ` (${Math.round(current.confidence * 100)}%)`
          : ""}
        .
      </p>
      {current.note && current.source === "archive" && (
        <details className="caption-details">
          <summary>Original AI caption — unverified</summary>
          <p>{current.note}</p>
          <small>
            Older captions may contain incorrect identities or behavior. They
            are not used as confirmed feeding, drinking or health records.
          </small>
        </details>
      )}
    </Modal>
  );
}

function ZoneEditor({ cameras, close, notify }: {cameras:Camera[];close:()=>void;notify:(message:string)=>void}) {
  const [cam, setCam] = useState(cameras[0]?.id || ""),
    [rect, setRect] = useState<[number,number,number,number]|null>(null),
    [start, setStart] = useState<[number,number]|null>(null),
    [name, setName] = useState(""),
    [kind, setKind] = useState("feeder"),
    [zones, setZones] = useState<Zone[]>([]),
    [busy, setBusy] = useState(false),
    [refresh, setRefresh] = useState(0),
    [reference, setReference] = useState<{camera:string;token:string;url:string;expires:number}|null>(null),
    [imageReady, setImageReady] = useState(false),
    [pictureRatio, setPictureRatio] = useState(16 / 9),
    [loading, setLoading] = useState(false),
    [pictureError, setPictureError] = useState("");
  const area = useRef<HTMLDivElement>(null);
  const canDraw = !!reference && reference.camera === cam && imageReady && !loading && !busy && reference.expires > Date.now();
  useEffect(() => {
    const controller = new AbortController();
    let objectUrl = "", timer:ReturnType<typeof setTimeout>|undefined;
    setReference(null); setImageReady(false); setRect(null); setStart(null);
    setPictureError(""); setLoading(true);
    async function freeze() {
      try {
        const response = await fetch(`/api/cameras/${encodeURIComponent(cam)}/zone-reference`, {
          method:"POST", signal:controller.signal,
        });
        if (!response.ok) {
          if (response.status === 401) window.dispatchEvent(new Event("locked"));
          const error = await response.json().catch(() => ({}));
          throw new Error(error.detail || "The screened picture is unavailable. Please refresh.");
        }
        const token = response.headers.get("X-Aviary-Zone-Reference");
        const ttl = Number(response.headers.get("X-Aviary-Zone-Ttl"));
        if (!token || !Number.isFinite(ttl) || ttl <= 0 || ttl > 300) throw new Error("Please refresh to load a new reference picture.");
        const expires = Date.now() + ttl * 1000;
        const image = await response.blob();
        if (controller.signal.aborted) return;
        objectUrl = URL.createObjectURL(image);
        setReference({camera:cam, token, url:objectUrl, expires});
        timer = setTimeout(() => {
          setReference(null); setImageReady(false); setRect(null); setStart(null);
          setPictureError("This picture has expired. Refresh and draw the zone again.");
        }, Math.max(0, expires - Date.now()));
      } catch (error) {
        if (!controller.signal.aborted) setPictureError(errorMessage(error));
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    if (cam) void freeze(); else setLoading(false);
    return () => {
      controller.abort(); clearTimeout(timer);
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [cam, refresh]);
  useEffect(() => {
    api<Zone[]>("/zones")
      .then(setZones)
      .catch((e) => notify(errorMessage(e)));
  }, []);
  function point(e:React.PointerEvent):[number,number] {
    const r = area.current!.getBoundingClientRect();
    return [
      Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)),
      Math.max(0, Math.min(1, (e.clientY - r.top) / r.height)),
    ];
  }
  async function save() {
    if (!rect || !name.trim() || !canDraw || !reference || start) return;
    setBusy(true);
    try {
      await api("/zones", {
        method: "POST",
        body: JSON.stringify({ camera: cam, name:name.trim(), kind, rect, reference_token:reference.token }),
      });
      setZones(await api<Zone[]>("/zones"));
      setRect(null);
      setName("");
      setRefresh(value => value + 1);
      notify("Zone saved. Visits will appear when a bird is tracked here.");
    } catch (e) {
      notify(errorMessage(e));
      setReference(null); setImageReady(false); setRect(null); setStart(null);
      setPictureError("Refresh the picture and draw the zone again before saving.");
    } finally {
      setBusy(false);
    }
  }
  return (
    <Modal title="Feeder & water zones" close={close} wide>
      <p className="muted">
        Load a screened picture, then draw a rectangle around a bowl or
        perch. Visits record proximity; they do not confirm eating or drinking.
        This exact picture is saved with the zone and checked before counting visits. Moving a camera or cage can pause counting until the reference view is recovered or you mark a new zone.
      </p>
      <div className="form-row">
        <label>
          Camera
          <select
            aria-label="Zone camera"
            value={cam}
            disabled={busy}
            onChange={(e) => {
              setCam(e.target.value);
              setRect(null);
            }}
          >
            {cameras.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Zone type
          <select aria-label="Zone type" value={kind} disabled={busy} onChange={(e) => setKind(e.target.value)}>
            <option value="feeder">Feeder</option>
            <option value="water">Water</option>
            <option value="perch">Resting perch</option>
            <option value="explore">Exploring area</option>
          </select>
        </label>
      </div>
      <button type="button" disabled={busy || loading || !cam} onClick={() => setRefresh(value => value + 1)}>
        <RefreshCw size={16}/> Refresh picture
      </button>
      <p className="muted" role="status">
        {loading ? "Loading a stable screened picture…" : pictureError || (imageReady ? "Frozen picture ready. Draw and save within five minutes." : "Loading picture…")}
      </p>
      <div
        className="zone-canvas"
        aria-label="Zone drawing canvas"
        aria-busy={loading}
        style={{aspectRatio:pictureRatio}}
        ref={area}
        onPointerDown={(e) => {
          if (!canDraw || !e.isPrimary || e.button !== 0) return;
          e.currentTarget.setPointerCapture(e.pointerId);
          setStart(point(e));
          setRect(null);
        }}
        onPointerMove={(e) => {
          if (!start || !canDraw) return;
          const p = point(e);
          setRect([
            Math.min(start[0], p[0]),
            Math.min(start[1], p[1]),
            Math.abs(start[0] - p[0]),
            Math.abs(start[1] - p[1]),
          ]);
        }}
        onPointerUp={() => setStart(null)}
        onPointerCancel={() => {setStart(null); setRect(null);}}
      >
        {reference?.camera === cam && <img
          key={reference.url}
          draggable={false}
          src={reference.url}
          alt="Frozen screened picture for this zone"
          onLoad={(event) => {setPictureRatio(event.currentTarget.naturalWidth / event.currentTarget.naturalHeight); setImageReady(true);}}
          onError={() => {setImageReady(false); setRect(null); setPictureError("This picture could not load. Please refresh.");}}
        />}
        {zones
          .filter((z) => imageReady && reference?.camera === cam && z.camera === cam)
          .map((z) => (
            <div
              key={z.id}
              className="zone saved-zone"
              style={{
                left: z.rect[0] * 100 + "%",
                top: z.rect[1] * 100 + "%",
                width: z.rect[2] * 100 + "%",
                height: z.rect[3] * 100 + "%",
              }}
            >
              <span>{z.name}</span>
            </div>
          ))}
        {rect && (
          <div
            className="zone"
            style={{
              left: rect[0] * 100 + "%",
              top: rect[1] * 100 + "%",
              width: rect[2] * 100 + "%",
              height: rect[3] * 100 + "%",
            }}
          />
        )}
      </div>
      <div className="form-row">
        <label>
          Zone name
          <input
            value={name}
            disabled={busy}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Front water bowl"
            maxLength={60}
          />
        </label>
        <button
          className="primary"
          disabled={!rect || rect[2] <= .005 || rect[3] <= .005 || !name.trim() || !canDraw || !!start}
          onClick={save}
        >
          <Plus size={18} />
          Save Zone
        </button>
      </div>
      <div className="zone-list">
        {zones
          .filter((z) => z.camera === cam)
          .map((z) => (
            <div key={z.id}>
              <MapPin size={16} />
              <span>{z.name}</span>
              <small>{z.kind}</small>
              <IconButton
                label={`Disable ${z.name}`}
                onClick={async () => {
                  await api("/zones/" + z.id, { method: "DELETE" });
                  setZones(zones.filter((x) => x.id !== z.id));
                }}
              >
                <X size={16} />
              </IconButton>
            </div>
          ))}
      </div>
    </Modal>
  );
}

function Ask({ birds, open, notify }: {birds:BirdProfile[];open:(event:Observation,items?:Observation[])=>void;notify:(message:string)=>void}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]),
    [text, setText] = useState(""),
    [busy, setBusy] = useState(false),
    [local, setLocal] = useState(false);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => {
    api<ChatMessage[]>("/chat")
      .then(setMessages)
      .catch((e) => notify(errorMessage(e)));
  }, []);
  useEffect(
    () => end.current?.scrollIntoView({ behavior: "smooth", block: "end" }),
    [messages, busy],
  );
  async function send(value = text) {
    if (!value.trim() || busy) return;
    setText("");
    setBusy(true);
    setMessages((m) => [...m, { role: "user", text: value }]);
    try {
      const response = await api<ChatMessage>("/ask", {
        method: "POST",
        body: JSON.stringify({ text: value, local_analysis: local }),
      });
      setMessages((m) => [...m, response]);
    } catch (e) {
      setMessages((m) => [
        ...m,
        { role: "assistant", text: errorMessage(e), evidence: [] },
      ]);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="ask-page">
      <div className="conversation">
        {!messages.length && (
          <div className="ask-welcome">
            <div className="ask-symbol">
              <MessageCircle size={32} />
            </div>
            <h2>Ask about the birds.</h2>
            <p>
              Ask about a day, a bird, or a moment you missed.
              <br />
              Answers stay grounded in your recorded sightings.
            </p>
            <div className="suggestions">
              {[
                "What happened today?",
                `When was ${birds[0]?.name || "my bird"} last seen?`,
                "Did everyone drink today?",
              ].map((q) => (
                <button key={q} onClick={() => send(q)}>
                  {q}
                  <ArrowUpRight size={16} />
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div className={"message " + m.role} key={i}>
            {m.role === "assistant" && <img src="/icon.svg" alt="Aviary" />}
            <div>
              <div className="bubble">{m.text}</div>
              {m.evidence && m.evidence.length > 0 && (
                <div className="evidence-strip">
                  {m.evidence.slice(0, 6).map((e) => (
                    <button key={e.id} onClick={() => open(e, m.evidence)}>
                      <Photo src={e.image} alt={e.bird} loading="lazy" />
                      <strong>{title(e.bird)}</strong>
                      <small>{stamp(e.time)}</small>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        {busy && (
          <div className="thinking">
            <span />
            <span />
            <span />
            {local
              ? "Reading with your local model…"
              : "Looking through your memories…"}
          </div>
        )}
        <div ref={end} />
      </div>
      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <div>
          <input
            aria-label="Ask about your birds"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="What have the birds been up to?"
            maxLength={2000}
          />
          <button
            className="send"
            disabled={!text.trim() || busy}
            aria-label="Send question"
          >
            <Send size={19} />
          </button>
        </div>
        <label className="local-choice">
          <input
            type="checkbox"
            checked={local}
            onChange={(e) => setLocal(e.target.checked)}
          />
          Add a local AI interpretation <span>May take a moment</span>
        </label>
      </form>
    </div>
  );
}

export default function App() {
  const [auth, setAuth] = useState<boolean|null>(null),
    [overview, setOverview] = useState<Overview|null>(null),
    [tab, setTab] = useState(
      new URLSearchParams(location.search).get("tab") || "live",
    ),
    [birdFilter, setBirdFilter] = useState(
      new URLSearchParams(location.search).get("bird") || "",
    ),
    [date, setDate] = useState(""),
    [search, setSearch] = useState(""),
    [saved, setSaved] = useState(false),
    [events, setEvents] = useState<Observation[]>([]),
    [cursor, setCursor] = useState<string|null>(null),
    [loading, setLoading] = useState(false),
    [selected, setSelected] = useState<Observation|null>(null),
    [replay, setReplay] = useState<Observation[]|null>(null),
    [settings, setSettings] = useState(false),
    [zones, setZones] = useState(false),
    [toast, setToast] = useState(""),
    [error, setError] = useState(""),
    [camera, setCamera] = useState("");
  const requestId = useRef(0);
  const swipeStart=useRef<{x:number;y:number}|null>(null);
  const birds = overview?.birds || [],
    cameras = overview?.cameras || [];
  function notify(message:string) {
    setToast(message);
    setTimeout(() => setToast(""), 6000);
  }
  useEffect(() => {
    api<{authenticated:boolean}>("/session")
      .then((x) => setAuth(x.authenticated))
      .catch(() => setAuth(false));
    const lock = () => setAuth(false);
    window.addEventListener("locked", lock);
    if ("serviceWorker" in navigator)
      navigator.serviceWorker.register("/sw.js").catch(() => {});
    return () => window.removeEventListener("locked", lock);
  }, []);
  useEffect(() => {
    if (!auth) return;
    let active = true;
    async function update() {
      try {
        const data = await api<Overview>("/overview");
        if (active) {
          setOverview(data);
          setError("");
        }
      } catch (e) {
        if (active) setError(errorMessage(e));
      }
    }
    update();
    const timer = setInterval(() => {
      if (!document.hidden) update();
    }, 5000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [auth]);
  useEffect(() => {
    const params = new URLSearchParams();
    if (tab !== "today") params.set("tab", tab);
    if (birdFilter) params.set("bird", birdFilter);
    history.replaceState(
      null,
      "",
      location.pathname + (params.size ? "?" + params : ""),
    );
  }, [tab, birdFilter]);
  async function load(more = false) {
    if (!auth) return;
    const id = ++requestId.current;
    setLoading(true);
    const params = new URLSearchParams({
      limit: tab === "today" ? "9" : "24",
      bird: birdFilter,
      date,
      search,
      favorites: String(saved),
    });
    if (more && cursor) params.set("before", cursor);
    try {
      const data = await api<{events:Observation[];next:string|null}>("/events?" + params);
      if (id === requestId.current) {
        setEvents((old) => (more ? [...old, ...data.events] : data.events));
        setCursor(data.next);
      }
    } catch (e) {
      notify(errorMessage(e));
    } finally {
      if (id === requestId.current) setLoading(false);
    }
  }
  useEffect(() => {
    if (["today", "birds", "memories"].includes(tab)) {
      const timer = setTimeout(() => load(), search ? 300 : 0);
      return () => clearTimeout(timer);
    }
  }, [auth, tab, birdFilter, date, search, saved]);
  function navigate(next:string, bird = "") {
    setTab(next);
    setBirdFilter(bird);
    setDate("");
    setSearch("");
    setSaved(false);
    window.scrollTo(0, 0);
  }
  function startSwipe(e:React.TouchEvent){
    if(!['live','clips','today'].includes(tab)||selected||settings||zones||e.touches.length!==1||(e.target as Element).closest('button,input,select,textarea,[data-no-swipe]')){swipeStart.current=null;return;}
    swipeStart.current={x:e.touches[0].clientX,y:e.touches[0].clientY};
  }
  function endSwipe(e:React.TouchEvent){
    const start=swipeStart.current;swipeStart.current=null;if(!start)return;
    const dx=e.changedTouches[0].clientX-start.x,dy=e.changedTouches[0].clientY-start.y;
    if(Math.abs(dx)<75||Math.abs(dx)<Math.abs(dy)*1.5)return;
    const order=['live','clips','today'],next=order.indexOf(tab)+(dx>0?1:-1);
    if(next>=0&&next<order.length)navigate(order[next]);
  }
  function open(event:Observation, items:Observation[] = events) {
    setSelected(event);
    setReplay(items);
  }
  async function favorite(event:Observation) {
    try {
      const x = await api<{favorite:boolean}>("/events/" + event.id + "/favorite", {
        method: "POST",
      });
      setEvents((old) =>
        old.map((e) =>
          e.id === event.id ? { ...e, favorite: x.favorite } : e,
        ),
      );
      if (replay)
        setReplay((old) =>
          (old || []).map((e) =>
            e.id === event.id ? { ...e, favorite: x.favorite } : e,
          ),
        );
      notify(
        x.favorite ? "Saved to your memories." : "Removed from saved memories.",
      );
    } catch (e) {
      notify(errorMessage(e));
    }
  }
  if (auth === null)
    return (
      <div className="boot">
        <img src="/icon.svg" alt="Aviary" />
        <p>Opening your aviary…</p>
      </div>
    );
  if (!auth) return <Login done={() => setAuth(true)} />;
  const currentCamera = cameras.find((c) => c.id === camera) || cameras[0];
  const activeBird = birds.find((b) => b.id === birdFilter);
  const heading =
    {
      today: "Feed",
      live: "Cameras",
      birds: activeBird ? activeBird.name : "The birds",
      memories: "Photo archive",
      clips: "Reels",
      stats: "Stats",
      review: "Clip review",
      learn: "Daily five",
      ask: "Ask Aviary",
    }[tab] || "Your aviary";
  return (
    <div className={'app watch-app '+(tab==='live'?'immersive-live':tab==='clips'?'immersive-reels':'')} onTouchStart={startSwipe} onTouchEnd={endSwipe}><UpdateNotice/>
      <aside className="sidebar">
        <button className="brand" onClick={() => navigate("today")}>
          <img src="/icon.svg" alt="" />
          <span>
            aviary<small>AT HOME</small>
          </span>
        </button>
        <nav aria-label="Main navigation">
          {tabs.map(([id, label, Icon]) => (
            <button
              key={id}
              aria-current={tab === id ? "page" : undefined}
              className={tab === id ? "active" : ""}
              onClick={() => navigate(id)}
            >
              <Icon size={20} />
              <span>{label}</span>
              {tab === id && <i />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="home-mark">
            <span className="dot on" />
            YOUR PRIVATE AVIARY
          </div>
          <button onClick={() => setSettings(true)}>
            <Settings size={19} />
            <span>Home Settings</span>
          </button>
        </div>
      </aside>
      <main className="main">
        <header className="page-header">
          <div>
            <div className="eyebrow">
              {new Date().toLocaleDateString(undefined, {
                weekday: "long",
                month: "long",
                day: "numeric",
              })}
            </div>
            <h1>{tab==='live'?'aviary':heading}</h1>
          </div>
          <div className="header-actions">
            <IconButton label="Daily five tagging" onClick={() => navigate('learn')}><Check size={19}/></IconButton>
            <IconButton label="Aviary menu" onClick={() => setSettings(true)}>
              <Settings size={20} />
            </IconButton>
          </div>
        </header>
        {error && (
          <div className="error-banner">
            {error}
            <button onClick={() => location.reload()}>Reconnect</button>
          </div>
        )}
        {!overview ? (
          <div className="skeleton-grid">
            <div />
            <div />
            <div />
          </div>
        ) : (
          <>
            {tab === "today" && <ActivityFeed cameras={cameras} birds={birds} open={open}/>}
            {tab === 'clips' && <VideoFeed reels cameras={cameras} notify={notify}/>}
            {tab === 'review' && <VideoFeed review cameras={cameras} notify={notify}/>}
            {tab === 'stats' && <NerdStats/>}
            {tab === 'learn' && <DailyTag cameras={cameras} back={()=>navigate('live')}/>}
            {tab === "live" && <LiveWall cameras={cameras} zones={()=>setZones(true)}/>}
            {tab === "birds" && (
              <>
                <div className="bird-switcher">
                  {birds.map((b) => (
                    <button
                      key={b.id}
                      className={b.id === birdFilter ? "selected" : ""}
                      onClick={() => setBirdFilter(b.id)}
                    >
                      <Avatar bird={b} size="tiny" />
                      {b.name}
                    </button>
                  ))}
                  <button
                    onClick={() => setBirdFilter("")}
                    className={!birdFilter ? "selected" : ""}
                  >
                    Everyone
                  </button>
                </div>
                {activeBird ? (
                  <section className="bird-profile">
                    <Avatar bird={activeBird} size="large" />
                    <div>
                      <div className="eyebrow">
                        {activeBird.species}{" "}
                        {activeBird.sex ? "· " + activeBird.sex : ""}
                      </div>
                      <h2>Hello, {activeBird.name}.</h2>
                      <p>
                        Last recorded {ago(activeBird.last?.time).toLowerCase()}
                        .
                      </p>
                    </div>
                    <div className="bird-stats">
                      <div>
                        <Eye size={18} />
                        <strong>{activeBird.today.toLocaleString()}</strong>
                        <small>Sightings today</small>
                      </div>
                      <div>
                        <Utensils size={18} />
                        <strong>
                          {activeBird.visits.feeder
                            ? Math.round(activeBird.visits.feeder) + "s"
                            : "Unknown"}
                        </strong>
                        <small>Feeder visits</small>
                      </div>
                      <div>
                        <Droplets size={18} />
                        <strong>
                          {activeBird.visits.water
                            ? Math.round(activeBird.visits.water) + "s"
                            : "Unknown"}
                        </strong>
                        <small>Water visits</small>
                      </div>
                    </div>
                  </section>
                ) : (
                  <div className="bird-directory">
                    {birds.map((b) => (
                      <button key={b.id} onClick={() => setBirdFilter(b.id)}>
                        <Avatar bird={b} size="large" />
                        <h2>{b.name}</h2>
                        <p>{b.species}</p>
                        <span>
                          Get to know {b.name}
                          <ArrowUpRight size={16} />
                        </span>
                      </button>
                    ))}
                  </div>
                )}
                <div className="section-heading">
                  <h2>
                    {activeBird
                      ? `${activeBird.name}’s moments`
                      : "Around the aviary"}
                  </h2>
                  <button
                    className="text-button"
                    disabled={!events.length}
                    onClick={() => open(events[0])}
                  >
                    <Play size={16} />
                    Replay moments
                  </button>
                </div>
                <VideoFeed bird={birdFilter} cameras={cameras} notify={notify}/><h3 className="archive-label">Photo archive</h3><div className="event-grid">
                  {events.map((e) => (
                    <EventCard
                      key={e.id}
                      event={e}
                      birds={birds}
                      cameras={cameras}
                      open={open}
                      onFavorite={favorite}
                    />
                  ))}
                </div>
              </>
            )}
            {tab === "memories" && (
              <>
                <div className="memory-toolbar">
                  <div className="search-box">
                    <Search size={18} />
                    <input
                      aria-label="Search memories"
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      placeholder="Search the archive…"
                    />
                  </div>
                  <select
                    aria-label="Filter by bird"
                    value={birdFilter}
                    onChange={(e) => setBirdFilter(e.target.value)}
                  >
                    <option value="">All birds</option>
                    {birds.map((b) => (
                      <option key={b.id} value={b.id}>
                        {b.name}
                      </option>
                    ))}
                  </select>
                  <label className="date-filter">
                    <CalendarDays size={17} />
                    <input
                      aria-label="Filter by date"
                      type="date"
                      value={date}
                      onChange={(e) => setDate(e.target.value)}
                    />
                  </label>
                  <button
                    className={"secondary " + (saved ? "selected" : "")}
                    onClick={() => setSaved(!saved)}
                  >
                    <Heart size={17} />
                    Saved
                  </button>
                </div>
                <div className="archive-caption">
                  <p>
                    {overview.archive.count.toLocaleString()} recorded sightings
                    to wander through.
                  </p>
                  {(birdFilter || date || search || saved) && (
                    <button
                      className="text-button"
                      onClick={() => {
                        setBirdFilter("");
                        setDate("");
                        setSearch("");
                        setSaved(false);
                      }}
                    >
                      Clear filters
                      <X size={15} />
                    </button>
                  )}
                </div>
                <div className="event-grid">
                  {events.map((e) => (
                    <EventCard
                      key={e.id}
                      event={e}
                      birds={birds}
                      cameras={cameras}
                      open={open}
                      onFavorite={favorite}
                    />
                  ))}
                </div>
                {!events.length && !loading && (
                  <Empty icon={Images} title="No moments here yet">
                    Try another day or bird, or clear your filters.
                  </Empty>
                )}
              </>
            )}
            {["memories", "birds"].includes(tab) && (
              <div className="load-more">
                {cursor && (
                  <button
                    className="secondary"
                    disabled={loading}
                    onClick={() => load(true)}
                  >
                    {loading
                      ? "Opening more moments…"
                      : "A little further back"}
                    <ArrowRight size={17} />
                  </button>
                )}
              </div>
            )}
            {tab === "ask" && <Ask birds={birds} open={open} notify={notify} />}
          </>
        )}
        <footer className="page-footer">
          <img src="/icon.svg" alt="" />
          <span>AVIARY / LOCAL</span>
        </footer>
      </main>
      <nav className="mobile-nav" aria-label="Mobile navigation">
        {tabs.map(([id, label, Icon]) => (
          <button
            className={tab === id ? "active" : ""}
            aria-current={tab === id ? "page" : undefined}
            key={id}
            onClick={() => navigate(id)}
          >
            <Icon size={20} />
            <span>{label}</span>
          </button>
        ))}
      </nav>
      {selected && (
        <Replay
          event={selected}
          events={replay || events}
          birds={birds}
          cameras={cameras}
          onFavorite={favorite}
          close={() => setSelected(null)}
        />
      )}{" "}
      {zones && (
        <ZoneEditor
          cameras={cameras}
          close={() => setZones(false)}
          notify={notify}
        />
      )}{" "}
      {settings && (
        <Modal title="Aviary" close={() => setSettings(false)}>
          <div className="settings-shortcuts"><button onClick={()=>{setSettings(false);navigate('learn')}}><Check size={20}/>Daily five tagging<ArrowRight size={16}/></button><button onClick={()=>{setSettings(false);navigate('birds')}}><Bird size={20}/>The birds<ArrowRight size={16}/></button><button onClick={()=>{setSettings(false);navigate('ask')}}><MessageCircle size={20}/>Ask Aviary<ArrowRight size={16}/></button><button onClick={()=>{setSettings(false);navigate('stats')}}><Activity size={20}/>Nerdy Stats<ArrowRight size={16}/></button><button onClick={()=>{setSettings(false);navigate('review')}}><Check size={20}/>Review Clips<ArrowRight size={16}/></button><button onClick={()=>{setSettings(false);navigate('memories')}}><Images size={20}/>Photo Archive<ArrowRight size={16}/></button></div>
          <div className="setting-block">
            <Download size={24} />
            <h3>Aviary on your home screen</h3>
            <p>
              On your iPad or iPhone, open this private address in Safari. Tap
              Share, then Add to Home Screen. Aviary opens in its own
              full-screen window.
            </p>
            <small>Keep Tailscale connected when you visit.</small>
          </div>
          <div className="setting-block">
            <MapPin size={24} />
            <h3>Little daily routines</h3>
            <p>
              Mark the feeder, water and favorite perches. Only recorded visits
              appear in the birds’ profiles.
            </p>
            <button
              className="secondary"
              onClick={() => {
                setSettings(false);
                setZones(true);
              }}
            >
              Set Up Zones
              <ArrowRight size={16} />
            </button>
          </div>
          <details className="setting-block">
            <summary>About your views & memories</summary>
            <p>
              Your camera video, photos and AI processing stay on your own
              machines. Smooth Live is original video and may show people. Older
              archive images have not been screened for people. AI captions and
              bird identities can be mistaken.
            </p>
            <p>
              3D positions require synchronized, calibrated cameras. Tagged
              zones store a reference image and, when available, a pan/tilt position. Visits pause when the saved view cannot be verified. Cameras without motors can move too.
            </p>
          </details>
          <button
            className="secondary"
            onClick={async () => {
              await api("/logout", { method: "POST" });
              setSettings(false);
              setAuth(false);
            }}
          >
            <LogOut size={17} />
            Lock This Device
          </button>
        </Modal>
      )}
      {toast && (
        <div className="toast" role="status">
          <Check size={17} />
          {toast}
        </div>
      )}
    </div>
  );
}
