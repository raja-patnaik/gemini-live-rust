use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use clap::Parser;
use futures::{SinkExt, StreamExt};
use image::{codecs::jpeg::JpegEncoder, ColorType};
use parking_lot::Mutex;
use scrap::{Capturer, Display};
use serde::Serialize;
use serde_json::json;
use std::{
    env,
    io::Write,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
use tokio::time::interval;
use tokio_tungstenite::tungstenite::Message;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// Simple Rust streamer to Gemini Live API: microphone audio + screen JPEG frames.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Frames per second for screen capture (JPEG snapshotns).
    #[arg(long, default_value_t = 1)]
    fps: u32,

    /// Model resource name (must support Live API).
    /// Common: models/gemini-live-2.5-flash-preview
    #[arg(long, default_value = "models/gemini-live-2.5-flash-preview")]
    model: String,

    /// If set, do not stream audio.
    #[arg(long, default_value_t = false)]
    no_audio: bool,

    /// If set, do not stream screen.
    #[arg(long, default_value_t = false)]
    no_screen: bool,

    /// Response modality: TEXT or AUDIO (Live API requires choosing one).
    #[arg(long, default_value = "TEXT")]
    response_modality: String,
}

#[derive(Serialize)]
struct Blob<'a> {
    data: &'a str,
    #[serde(rename = "mimeType")]
    mime_type: &'a str,
}

/// Downsample nearest-neighbor from in_rate -> out_rate (mono).
fn downsample_i16_nearest(input: &[i16], in_rate: u32, out_rate: u32) -> Vec<i16> {
    if in_rate == out_rate || input.is_empty() {
        return input.to_vec();
    }
    let ratio = in_rate as f32 / out_rate as f32;
    let out_len = (input.len() as f32 / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    let mut pos = 0.0f32;
    for _ in 0..out_len {
        let idx = pos as usize;
        out.push(input[idx.min(input.len() - 1)]);
        pos += ratio;
    }
    out
}

/// Convert interleaved stereo to mono by averaging L/R.
fn stereo_to_mono_i16(input: &[i16]) -> Vec<i16> {
    let mut out = Vec::with_capacity(input.len() / 2);
    for s in input.chunks_exact(2) {
        let l = s[0] as i32;
        let r = s[1] as i32;
        out.push(((l + r) / 2) as i16);
    }
    out
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let api_key = env::var("GEMINI_API_KEY")
        .context("Set GEMINI_API_KEY in your environment (AI Studio API key)")?;

    // Compose Live API WebSocket URL with API key.
    // Docs: WebSocket endpoint; auth via ?key=... is supported. 
    let url = format!("wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={}", api_key);

    eprintln!("Connecting to Live API at: {}", url);
    let (ws_stream, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .context("WebSocket connect failed")?;

    let (mut ws_tx, mut ws_rx) = ws_stream.split();

    // 1) Send the initial setup message.
    //    You must choose one response modality per session: TEXT or AUDIO.
    let setup = json!({
      "setup": {
        "model": args.model,
        "generationConfig": {
          "responseModalities": [args.response_modality.to_uppercase()],
          // For vision inputs, you may specify mediaResolution; low helps latency.
          "mediaResolution": "MEDIA_RESOLUTION_LOW"
        },
        "systemInstruction": {
          "parts": [{
            "text": "You are a helpful assistant. I will stream my screen and audio; describe and summarize what you see and hear succinctly."
          }]
        },
        // Ask the server to transcribe the input audio while we stream.
        "inputAudioTranscription": {}
      }
    });

    ws_tx.send(Message::Text(setup.to_string())).await?;

    // 2) Wait for setupComplete before we start streaming.
    eprintln!("Waiting for setupComplete...");
    while let Some(msg) = ws_rx.next().await {
        let msg = msg?;
        if let Message::Text(txt) = msg {
            // Print any early messages for visibility.
            if txt.contains("setupComplete") {
                eprintln!("Session setup complete.\n");
                break;
            } else {
                eprintln!("(pre-setup message) {}", txt);
            }
        }
    }

    // Channel to serialize all outgoing realtimeInput messages through a single task.
    let (out_tx, mut out_rx) = mpsc::channel::<Message>(128);

    // 3) Spawn a writer task that pulls JSON messages from out_rx and writes to socket.
    let mut ws_writer = ws_tx;
    let writer = tokio::spawn(async move {
        while let Some(msg) = out_rx.recv().await {
            if let Err(e) = ws_writer.send(msg).await {
                eprintln!("WebSocket send error: {e}");
                break;
            }
        }
    });

    // 4) Stream AUDIO (microphone) if enabled.
    let audio_handle = if !args.no_audio {
        let sender = out_tx.clone();
        Some(tokio::spawn(async move {
            if let Err(e) = stream_mic_audio(sender).await {
                eprintln!("Audio streaming error: {e:?}");
            }
        }))
    } else {
        None
    };

    // 5) Stream SCREEN (JPEG frames) if enabled.
    let screen_handle = if !args.no_screen {
        let fps = args.fps.max(1);
        let sender = out_tx.clone();
        Some(tokio::spawn(async move {
            if let Err(e) = stream_screen_jpeg(sender, fps).await {
                eprintln!("Screen streaming error: {e:?}");
            }
        }))
    } else {
        None
    };

    // 6) Reader task: print server messages (text and transcriptions).
    let reader = tokio::spawn(async move {
        while let Some(msg) = ws_rx.next().await {
            match msg {
                Ok(Message::Text(txt)) => {
                    // Very lightweight parsing to surface useful info.
                    if txt.contains("\"text\"") {
                        println!("{txt}");
                    } else {
                        eprintln!("(server) {txt}");
                    }
                }
                Ok(Message::Binary(_b)) => {
                    eprintln!("(server) [binary message]");
                }
                Ok(Message::Close(c)) => {
                    eprintln!("Server closed: {:?}", c);
                    break;
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!("WebSocket read error: {e}");
                    break;
                }
            }
        }
    });

    // Keep running until Ctrl+C.
    tokio::signal::ctrl_c().await.ok();
    eprintln!("Shutting down...");

    drop(out_tx); // end writer

    if let Some(h) = audio_handle { let _ = h.await; }
    if let Some(h) = screen_handle { let _ = h.await; }

    let _ = writer.await;
    let _ = reader.await;

    Ok(())
}

/// Capture microphone using cpal, convert to PCM16 mono 16kHz, stream in ~200ms chunks.
async fn stream_mic_audio(out: mpsc::Sender<Message>) -> Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("No default input device")?;
    let default_cfg = device.default_input_config().context("No default input config")?;
    let in_rate = default_cfg.sample_rate().0;
    let in_channels = default_cfg.channels();

    eprintln!("Audio: device={} rate={}Hz channels={}", device.name().unwrap_or_default(), in_rate, in_channels);

    // Shared sample buffer; callback pushes raw i16 interleaved samples.
    let shared: Arc<Mutex<Vec<i16>>> = Arc::new(Mutex::new(Vec::with_capacity((in_rate as usize) * 2)));
    let shared_cb = shared.clone();

    // Build stream matching device's native sample format.
    let err_fn = |e| eprintln!("CPAL stream error: {e}");
    let stream = match default_cfg.sample_format() {
        cpal::SampleFormat::F32 => {
            let config: cpal::StreamConfig = default_cfg.clone().into();
            device.build_input_stream(
                &config,
                move |data: &[f32], _| {
                    let mut buf = shared_cb.lock();
                    // Downmix to mono later; push raw as i16 now
                    for &s in data {
                        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                        buf.push(v);
                    }
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::I16 => {
            let config: cpal::StreamConfig = default_cfg.clone().into();
            device.build_input_stream(
                &config,
                move |data: &[i16], _| {
                    shared_cb.lock().extend_from_slice(data);
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::U16 => {
            let config: cpal::StreamConfig = default_cfg.clone().into();
            device.build_input_stream(
                &config,
                move |data: &[u16], _| {
                    let mut buf = shared_cb.lock();
                    for &s in data {
                        // Map 0..65535 -> -32768..32767 safely, then cast to i16
                        let centered: i32 = s as i32 - 32768;
                        buf.push(centered as i16);
                    }
                },
                err_fn,
                None,
            )?
        }
        _ => {
            return Err(anyhow::anyhow!("Unsupported audio sample format"));
        }
    };

    stream.play()?;

    // Ship chunks ~200ms at a time.
    let out_rate = 16_000u32;
    let chunk_ms = 200u64;
    let mut ticker = interval(Duration::from_millis(chunk_ms));
    let samples_per_chunk = (out_rate as u64 * chunk_ms / 1000) as usize;

    loop {
        ticker.tick().await;

        // Take a snapshot of the current buffer.
        let raw: Vec<i16> = {
            let mut lock = shared.lock();
            if lock.is_empty() {
                continue;
            }
            let v = lock.split_off(0);
            v
        };

        // Convert to mono if necessary.
        let mono = if in_channels >= 2 {
            stereo_to_mono_i16(&raw)
        } else {
            raw
        };

        // Resample to 16kHz (nearest-neighbor for simplicity).
        let pcm16_16k = downsample_i16_nearest(&mono, in_rate, out_rate);

        // Chop into fixed chunks (~200ms each) so the server can consume steadily.
        let mut offset = 0usize;
        while offset + samples_per_chunk <= pcm16_16k.len() {
            let slice = &pcm16_16k[offset..offset + samples_per_chunk];
            offset += samples_per_chunk;

            // Encode little-endian PCM16 -> bytes -> base64.
            let mut bytes = Vec::with_capacity(slice.len() * 2);
            for s in slice {
                bytes.write_all(&s.to_le_bytes())?;
            }
            let b64 = BASE64.encode(&bytes);

            let msg = json!({
              "realtimeInput": {
                "audio": {
                  "data": b64,
                  "mimeType": "audio/pcm;rate=16000"
                }
              }
            });

            out.send(Message::Text(msg.to_string())).await.ok();
        }
    }
}

/// Capture screen with `scrap`, JPEG-encode frames, and stream them.
async fn stream_screen_jpeg(out: mpsc::Sender<Message>, fps: u32) -> Result<()> {
    let display = Display::primary().context("No primary display")?;
    let mut capturer = Capturer::new(display).context("Failed to start screen capture")?;
    let (w, h) = (capturer.width(), capturer.height());
    eprintln!("Screen: {}x{}, streaming ~{} fps as image/jpeg", w, h, fps);

    let mut tick = interval(Duration::from_millis(1000 / fps as u64));
    let mut last = Instant::now();

    loop {
        tick.tick().await;

        // Try to grab a new frame
        let frame = loop {
            match capturer.frame() {
                Ok(buffer) => break buffer,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // Frame not ready yet
                    tokio::time::sleep(Duration::from_millis(5)).await;
                    continue;
                }
                Err(e) => return Err(e.into()),
            }
        };

        // `scrap` gives BGRA bytes with a stride (bytes_per_row).
        let stride = frame.len() / h;
        let mut rgb = Vec::with_capacity(w * h * 3);

        for y in 0..h {
            let row = &frame[y * stride..(y + 1) * stride];
            // row pixels are BGRA; take every 4 bytes
            for px in row.chunks_exact(4).take(w) {
                let b = px[0];
                let g = px[1];
                let r = px[2];
                rgb.push(r);
                rgb.push(g);
                rgb.push(b);
            }
        }

        // JPEG encode with quality 80 to keep sizes sane.
        let mut jpeg = Vec::new();
        {
            let mut enc = JpegEncoder::new_with_quality(&mut jpeg, 80);
            enc.encode(&rgb, w as u32, h as u32, ColorType::Rgb8)?;
        }
        let b64 = BASE64.encode(&jpeg);

        let msg = json!({
          "realtimeInput": {
            "video": {
              "data": b64,
              // Streaming visuals as individual frames:
              // For a full video stream, use `video/webm` (VP8/VP9/H264) instead.
              "mimeType": "image/jpeg"
            }
          }
        });

        out.send(Message::Text(msg.to_string())).await.ok();

        let now = Instant::now();
        let dt = now.duration_since(last);
        last = now;
        eprintln!("sent frame ({} ms)", dt.as_millis());
    }
}