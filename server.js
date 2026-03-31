import express from "express";
import cors from "cors";
import { pipeline } from "@xenova/transformers";

// ─── Config ───────────────────────────────────────────────
const PORT = process.env.PORT || 3333;
const API_KEY = process.env.EMBEDDING_API_KEY || ""; // Optional: set to secure your endpoint
const MODEL_NAME = "Xenova/paraphrase-multilingual-MiniLM-L12-v2";
const MAX_INPUT_LENGTH = 8000;
const MAX_BATCH_SIZE = 100;

// ─── Model Singleton ──────────────────────────────────────
let embeddingPipeline = null;

async function loadModel() {
  console.log(`[Embedding] Loading model: ${MODEL_NAME}...`);
  const startTime = Date.now();

  embeddingPipeline = await pipeline("feature-extraction", MODEL_NAME);

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`[Embedding] Model loaded in ${elapsed}s (384-dim, local)`);
}

// ─── Embedding Logic ──────────────────────────────────────
async function generateEmbedding(text) {
  const sanitized = text.trim().slice(0, MAX_INPUT_LENGTH);

  const output = await embeddingPipeline(sanitized, {
    pooling: "mean",
    normalize: true,
  });

  const embedding = Array.from(output.data);

  if (!embedding || embedding.length === 0) {
    throw new Error("Empty model output");
  }

  return embedding;
}

// ─── Express App ──────────────────────────────────────────
const app = express();

app.use(cors());
app.use(express.json({ limit: "10mb" }));

// ─── Auth Middleware (optional) ───────────────────────────
function authMiddleware(req, res, next) {
  if (!API_KEY) return next(); // No key set = open access

  const provided =
    req.headers["x-api-key"] ||
    req.headers["authorization"]?.replace("Bearer ", "");

  if (provided !== API_KEY) {
    return res.status(401).json({ error: "Invalid API key" });
  }

  next();
}

// ─── Readiness check ─────────────────────────────────────
function modelReady(req, res, next) {
  if (!embeddingPipeline) {
    return res.status(503).json({ error: "Model is still loading, try again shortly" });
  }
  next();
}

// ─── Routes ──────────────────────────────────────────────

// Health check (no auth required)
app.get("/health", (req, res) => {
  res.json({
    status: embeddingPipeline ? "ready" : "loading",
    model: MODEL_NAME,
    dimensions: 384,
    uptime: process.uptime(),
  });
});

// Single text embedding
app.post("/embed", authMiddleware, modelReady, async (req, res) => {
  try {
    const { text } = req.body;

    if (!text || typeof text !== "string" || text.trim().length === 0) {
      return res.status(400).json({ error: "Field 'text' must be a non-empty string" });
    }

    const startTime = Date.now();
    const embedding = await generateEmbedding(text);
    const elapsed = Date.now() - startTime;

    res.json({ embedding, dimensions: embedding.length, latencyMs: elapsed });
  } catch (err) {
    console.error("[Embed] Error:", err.message);
    res.status(500).json({ error: "Embedding failed", details: err.message });
  }
});

// Batch embedding
app.post("/embed-batch", authMiddleware, modelReady, async (req, res) => {
  try {
    const { texts } = req.body;

    if (!Array.isArray(texts) || texts.length === 0) {
      return res.status(400).json({ error: "Field 'texts' must be a non-empty array of strings" });
    }

    if (texts.length > MAX_BATCH_SIZE) {
      return res.status(400).json({
        error: `Batch too large. Max ${MAX_BATCH_SIZE} texts per request.`,
      });
    }

    const startTime = Date.now();
    const embeddings = [];

    for (let i = 0; i < texts.length; i++) {
      if (!texts[i] || typeof texts[i] !== "string" || texts[i].trim().length === 0) {
        return res.status(400).json({ error: `texts[${i}] must be a non-empty string` });
      }
      const embedding = await generateEmbedding(texts[i]);
      embeddings.push(embedding);
    }

    const elapsed = Date.now() - startTime;

    res.json({
      embeddings,
      count: embeddings.length,
      dimensions: 384,
      latencyMs: elapsed,
    });
  } catch (err) {
    console.error("[Embed-Batch] Error:", err.message);
    res.status(500).json({ error: "Batch embedding failed", details: err.message });
  }
});

// ─── Start ───────────────────────────────────────────────
async function start() {
  // Load model BEFORE accepting requests
  await loadModel();

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[Embedding Service] Running on http://0.0.0.0:${PORT}`);
    console.log(`[Embedding Service] Endpoints:`);
    console.log(`  GET  /health`);
    console.log(`  POST /embed       { "text": "..." }`);
    console.log(`  POST /embed-batch { "texts": ["...", "..."] }`);
    if (API_KEY) {
      console.log(`[Embedding Service] API key protection: ENABLED`);
    } else {
      console.log(`[Embedding Service] API key protection: DISABLED (set EMBEDDING_API_KEY to enable)`);
    }
  });
}

start().catch((err) => {
  console.error("[Embedding Service] Fatal startup error:", err);
  process.exit(1);
});

// Graceful shutdown
process.on("SIGTERM", () => {
  console.log("[Embedding Service] Shutting down...");
  process.exit(0);
});

process.on("SIGINT", () => {
  console.log("[Embedding Service] Shutting down...");
  process.exit(0);
});
