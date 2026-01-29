// server.js

// Load environment variables from .env file
require('dotenv').config();

const express = require('express');
const cors = require('cors');
const path = require('path');
// ⚠️ NOTE: We use GoogleGenAI for general chat/analysis and structured output.
// The TTS and Image generation parts of the frontend are using direct REST API calls 
// which are more complex to proxy securely in a single server function without specific libraries.
const { GoogleGenAI } = require('@google/genai');

// --- SERVER CONFIGURATION ---
const app = express();
const port = process.env.PORT || 3000;
const API_KEY = process.env.GEMINI_API_KEY;

// Serve static files from the current directory
app.use(express.static(path.join(__dirname)));

// 🛑 IMPORTANT: Configure CORS to allow requests from your frontend origins
app.use(cors({
    origin: [
        'http://127.0.0.1:5501', // Local development
        'http://localhost:3000',  // Local development
        'http://localhost:5500',  // VS Code Live Server
        /\.vercel\.app$/, // Allow all Vercel deployments
        process.env.FRONTEND_URL // Allow custom frontend URL via env variable
    ].filter(Boolean)
}));

app.use(express.json({ limit: '50mb' })); // Allows server to read large JSON/Base664 bodies (for images)

// --- GEMINI INITIALIZATION ---
if (!API_KEY) {
    console.error("FATAL ERROR: GEMINI_API_KEY is not set in environment variables (.env file).");
    process.exit(1);
}
const ai = new GoogleGenAI({}); 


// =================================================================
// 🏠 ROOT ROUTE: Serve the main HTML file
// =================================================================
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'markrt.html'));
});


// =================================================================
// 🚀 DEDICATED ENDPOINT 1: BASIC/MULTIMODAL CHAT (Text, Grounding, Image/Video)
// URL: /api/chat
// =================================================================
app.post('/api/chat', async (req, res) => {
    try {
        const { prompt, file } = req.body; // 'file' includes base64Data and mimeType

        if (!prompt) {
            return res.status(400).json({ error: 'Prompt is required.' });
        }

        const contents = [];
        const parts = [];

        // 1. Add File Part (Multimodal)
        if (file) {
            console.log(`Processing file type: ${file.mimeType}`);
            parts.push({
                inlineData: {
                    data: file.base64Data,
                    mimeType: file.mimeType
                }
            });
        }

        // 2. Add Text Part
        parts.push({ text: prompt });

        contents.push({ role: "user", parts: parts });

        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: contents,
            config: {
                // Enable Google Search Grounding for current market data
                tools: [{ googleSearch: {} }] 
            }
        });

        const citationSources = response.candidates?.[0]?.groundingMetadata?.webSearchQueries?.map(query => ({
            uri: `https://www.google.com/search?q=${encodeURIComponent(query)}`,
            title: query // Display the search query as the title
        })) || [];


        res.json({
            text: response.text,
            sources: citationSources
        });

    } catch (error) {
        console.error('API Error (/api/chat):', error);
        res.status(500).json({
            error: 'An internal server error occurred while analyzing the request.',
            details: error.message
        });
    }
});


// =================================================================
// 📈 DEDICATED ENDPOINT 2: STRUCTURED SENTIMENT ANALYSIS
// URL: /api/sentiment
// =================================================================
app.post('/api/sentiment', async (req, res) => {
    try {
        const { asset } = req.body;

        if (!asset) {
            return res.status(400).json({ error: 'Asset name is required for sentiment analysis.' });
        }

        const prompt = `Analyze the current market sentiment, key drivers, and outlook for the asset: **${asset}**. Search the web for real-time data and news.`;

        const SENTIMENT_SCHEMA = {
            type: "OBJECT",
            properties: {
                asset: { "type": "STRING", description: "The asset analyzed (e.g., BTC/USD)." },
                sentiment_score: { "type": "NUMBER", description: "A score from -10 (Extremely Bearish) to +10 (Extremely Bullish)."},
                key_drivers: {
                    "type": "ARRAY",
                    "items": { "type": "STRING", description: "Top 3 factors currently affecting the asset's price, e.g., 'Fed Rate Hike Speculation' or 'Major Exchange Hack'." }
                },
                summary: { "type": "STRING", description: "A concise paragraph summarizing the current market sentiment and outlook based on the key drivers." }
            },
            required: ["asset", "sentiment_score", "key_drivers", "summary"]
        };


        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: [{ role: "user", parts: [{ text: prompt }] }],
            config: {
                tools: [{ googleSearch: {} }],
                responseMimeType: "application/json",
                responseSchema: SENTIMENT_SCHEMA
            }
        });

        // The response text will be a JSON string that conforms to the schema
        const structuredData = JSON.parse(response.text);

        res.json(structuredData);

    } catch (error) {
        console.error('API Error (/api/sentiment):', error);
        // Attempt to parse the actual API response body for more detail if needed
        res.status(500).json({
            error: 'An internal server error occurred during structured sentiment analysis.',
            details: error.message
        });
    }
});


// =================================================================
// 🖼️ DEDICATED ENDPOINT 3: IMAGE GENERATION
// URL: /api/generate-image
// =================================================================
// NOTE: We directly proxy the Imagen REST API call here to handle the image generation flow.
app.post('/api/generate-image', async (req, res) => {
    try {
        const { prompt } = req.body;

        if (!prompt) {
            return res.status(400).json({ error: 'Image prompt is required.' });
        }
        
        // This is a proxy to the actual Imagen REST API (not the simpler @google/genai SDK format)
        const IMAGEN_API_URL = `https://us-central1-aiplatform.googleapis.com/v1/projects/YOUR_PROJECT_ID/locations/us-central1/publishers/google/models/imagen-3.0-generate-002:generateContent?key=${API_KEY}`;
        // ⚠️ You must replace YOUR_PROJECT_ID with your actual Google Cloud Project ID.
        // For simplicity and to fit the frontend, we'll try to use the GenAI SDK structure which works if 
        // the client is only expecting the base64 output.

        // Fallback/Simpler approach using the main AI SDK, which might work for simpler Image Generation models:
        // You should use the actual Imagen client library or the REST API for best results.
        const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:generateImages?key=${API_KEY}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                config: {
                    numberOfImages: 1,
                    aspectRatio: "1:1"
                }
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
             throw new Error(`Imagen API Failed: ${errorText}`);
        }

        const data = await response.json();
        
        // Return the first image's base64 data
        const base64Image = data.generatedImages[0].image.imageBytes;

        res.json({ base64Image: base64Image });

    } catch (error) {
        console.error('API Error (/api/generate-image):', error);
        res.status(500).json({
            error: 'An internal server error occurred during image generation.',
            details: error.message
        });
    }
});

// =================================================================
// 🔊 DEDICATED ENDPOINT 4: TEXT-TO-SPEECH (TTS)
// URL: /api/tts
// =================================================================
// NOTE: We proxy the TTS REST API call here. The frontend expects raw audio data.
app.post('/api/tts', async (req, res) => {
    try {
        const { text } = req.body;

        if (!text) {
            return res.status(400).json({ error: 'Text is required for TTS.' });
        }

        const TTS_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key=${API_KEY}`;
        
        const payload = {
            contents: [{ parts: [{ text: text }] }],
            generationConfig: {
                responseModalities: ["AUDIO"],
                speechConfig: {
                    voiceConfig: {
                        prebuiltVoiceConfig: { voiceName: "Charon" }
                    }
                }
            }
        };

        const response = await fetch(TTS_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`TTS API Failed: ${errorText}`);
        }

        const result = await response.json();

        const part = result?.candidates?.[0]?.content?.parts?.[0];
        
        if (!part || !part.inlineData || !part.inlineData.data) {
            throw new Error("TTS response missing inline audio data.");
        }

        // Return the raw base64 audio data and mime type
        res.json({
            audioData: part.inlineData.data,
            mimeType: part.inlineData.mimeType
        });

    } catch (error) {
        console.error('API Error (/api/tts):', error);
        res.status(500).json({
            error: 'An internal server error occurred during Text-to-Speech generation.',
            details: error.message
        });
    }
});


// --- SERVER STARTUP ---
app.listen(port, () => {
    console.log(`✅ Backend server listening at http://localhost:${port}`);
    console.log('Ensure this server stays running while you use your web page.');
});