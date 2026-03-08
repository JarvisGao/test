import dotenv from 'dotenv';
dotenv.config({ override: true });
import { GoogleGenAI } from '@google/genai';
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
async function run() {
  try {
    const res = await ai.models.embedContent({
      model: 'text-embedding-004',
      contents: 'hello',
      config: {
        httpOptions: {
          apiVersion: 'v1alpha'
        }
      }
    });
    console.log('text-embedding-004 success', res.embeddings?.[0]?.values?.length);
  } catch (e) {
    console.error('text-embedding-004 failed', e);
  }
}
run();
