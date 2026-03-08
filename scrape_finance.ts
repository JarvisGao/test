import axios from 'axios';
import * as cheerio from 'cheerio';
import db from './src/db.js';
import { getEmbedding } from './src/agents/task2.js';

async function scrapeFinance() {
  console.log('Starting to scrape financial knowledge...');
  
  const urls = [
    'https://zh.wikipedia.org/wiki/%E9%87%91%E8%9E%8D',
    'https://zh.wikipedia.org/wiki/%E8%82%A1%E7%A5%A8',
    'https://zh.wikipedia.org/wiki/%E5%80%BA%E5%88%B8',
    'https://zh.wikipedia.org/wiki/%E5%9F%BA%E9%87%91',
    'https://zh.wikipedia.org/wiki/%E8%A1%8D%E7%94%9F%E6%80%A7%E9%87%91%E8%9E%8D%E5%95%86%E5%93%81',
    'https://zh.wikipedia.org/wiki/%E5%A4%96%E6%B1%87%E5%B8%82%E5%9C%BA',
    'https://zh.wikipedia.org/wiki/%E5%AE%8F%E8%A7%82%E7%BB%8F%E6%B5%8E%E5%AD%A6',
    'https://zh.wikipedia.org/wiki/%E5%BE%AE%E8%A7%82%E7%BB%8F%E6%B5%8E%E5%AD%A6'
  ];
  
  let totalInserted = 0;
  const insertStmt = db.prepare('INSERT INTO documents (content, metadata, embedding) VALUES (?, ?, ?)');

  for (const url of urls) {
    try {
      console.log(`Scraping ${url}...`);
      const response = await axios.get(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
      });
      const $ = cheerio.load(response.data);
      
      const paragraphs: string[] = [];
      
      // Extract text from paragraphs
      $('p').each((i, el) => {
        const text = $(el).text().trim();
        if (text.length > 50 && !text.includes('维基百科') && !text.includes('Wikipedia')) { // Only keep meaningful paragraphs
          paragraphs.push(text);
        }
      });

      console.log(`Extracted ${paragraphs.length} paragraphs from ${url}. Processing...`);

      // Process up to 15 paragraphs per URL to get a good mix
      for (const p of paragraphs.slice(0, 15)) {
        try {
          const embedding = await getEmbedding(p);
          insertStmt.run(
            p,
            JSON.stringify({ source: url, type: 'scraped_knowledge' }),
            JSON.stringify(embedding)
          );
          totalInserted++;
          if (totalInserted % 10 === 0) {
            console.log(`Inserted ${totalInserted} paragraphs so far...`);
          }
        } catch (err) {
          console.error('Error inserting paragraph:', err);
        }
      }
    } catch (error) {
      console.error(`Error scraping data from ${url}:`, error);
    }
  }
  
  console.log(`Successfully scraped and inserted ${totalInserted} items into the knowledge base.`);
}

scrapeFinance();
