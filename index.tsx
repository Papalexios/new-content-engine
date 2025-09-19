import { GoogleGenerativeAI as GoogleGenAI } from "@google/generative-ai";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import React, { useState, useMemo, useEffect, useCallback, useReducer, useRef, memo } from 'react';
import ReactDOM from 'react-dom/client';

const AI_MODELS = {
    GEMINI_FLASH: 'gemini-2.5-flash',
    GEMINI_IMAGEN: 'imagen-4.0-generate-001',
    OPENAI_GPT4_TURBO: 'gpt-4-turbo',
    OPENAI_DALLE3: 'dall-e-3',
    ANTHROPIC_OPUS: 'claude-3-opus-20240229',
    ANTHROPIC_HAIKU: 'claude-3-haiku-20240307',
    OPENROUTER_DEFAULT: [
        'google/gemini-2.5-flash',
        'anthropic/claude-3-haiku',
        'microsoft/wizardlm-2-8x22b',
        'openrouter/auto'
    ],
    GROQ_MODELS: [
        'llama-3.1-70b-versatile',
        'llama-3.1-8b-instant',
        'gemma2-9b-it',
        'llama3-70b-8192',
        'llama3-8b-8192',
        'mixtral-8x7b-32768',
        'gemma-7b-it',
        'meta-llama/llama-4-scout-17b-16e-instruct',
    ]
};

// --- START: Performance & Caching Enhancements ---

/**
 * A sophisticated caching layer for API responses to reduce redundant calls
 * and improve performance within a session.
 */
class ContentCache {
  private cache = new Map<string, {data: any, timestamp: number}>();
  private TTL = 3600000; // 1 hour
  
  set(key: string, data: any) {
    this.cache.set(key, {data, timestamp: Date.now()});
  }
  
  get(key: string): any | null {
    const item = this.cache.get(key);
    if (item && Date.now() - item.timestamp < this.TTL) {
      console.log(`[Cache] HIT for key: ${key}`);
      return item.data;
    }
    console.log(`[Cache] MISS for key: ${key}`);
    return null;
  }
}
const apiCache = new ContentCache();

// --- END: Performance & Caching Enhancements ---


// --- START: Core Utility Functions ---

// Debounce function to limit how often a function gets called
const debounce = (func: (...args: any[]) => void, delay: number) => {
    let timeoutId: ReturnType<typeof setTimeout>;
    return (...args: any[]) => {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
            func.apply(null, args);
        }, delay);
    };
};


/**
 * A highly resilient function to extract a JSON object from a string.
 * It surgically finds the JSON boundaries by balancing brackets, strips conversational text and markdown,
 * and automatically repairs common syntax errors like trailing commas.
 * @param text The raw string response from the AI, which may contain conversational text.
 * @returns The clean, valid JSON object.
 * @throws {Error} if a valid JSON object cannot be found or parsed.
 */
const extractJson = (text: string): string => {
    if (!text || typeof text !== 'string') {
        throw new Error("Input text is invalid or empty.");
    }
    
    // First, try a simple parse. If it's valid, we're done.
    try {
        JSON.parse(text);
        return text;
    } catch (e) { /* Not valid, proceed with cleaning */ }

    // Aggressively clean up common conversational text and markdown fences.
    let cleanedText = text
        .replace(/^```(?:json)?\s*/, '') // Remove opening ```json or ```
        .replace(/\s*```$/, '')           // Remove closing ```
        .trim();

    // Find the first real start of a JSON object or array.
    const firstBracket = cleanedText.indexOf('{');
    const firstSquare = cleanedText.indexOf('[');
    
    if (firstBracket === -1 && firstSquare === -1) {
        console.error(`[extractJson] No JSON start characters ('{' or '[') found after cleanup.`, { originalText: text });
        throw new Error("No JSON object/array found in response.");
    }

    let startIndex = -1;
    if (firstBracket === -1) startIndex = firstSquare;
    else if (firstSquare === -1) startIndex = firstBracket;
    else startIndex = Math.min(firstBracket, firstSquare);

    let potentialJson = cleanedText.substring(startIndex);
    
    // Find the balanced end bracket for the structure.
    const startChar = potentialJson[0];
    const endChar = startChar === '{' ? '}' : ']';
    
    let balance = 1;
    let inString = false;
    let escapeNext = false;
    let endIndex = -1;

    for (let i = 1; i < potentialJson.length; i++) {
        const char = potentialJson[i];
        
        if (escapeNext) {
            escapeNext = false;
            continue;
        }
        
        if (char === '\\') {
            escapeNext = true;
            continue;
        }
        
        if (char === '"' && !escapeNext) {
            inString = !inString;
        }
        
        if (inString) continue;

        if (char === startChar) balance++;
        else if (char === endChar) balance--;

        if (balance === 0) {
            endIndex = i;
            break;
        }
    }

    let jsonCandidate;
    if (endIndex !== -1) {
        jsonCandidate = potentialJson.substring(0, endIndex + 1);
    } else {
        jsonCandidate = potentialJson; // Truncated response, try to parse what we have.
        console.warn("[extractJson] Could not find a balanced closing bracket. The AI response may have been truncated.");
    }

    // Attempt to parse the candidate string.
    try {
        JSON.parse(jsonCandidate);
        return jsonCandidate;
    } catch (e) {
        // If parsing fails, try to repair common issues like trailing commas.
        console.warn("[extractJson] Initial parse failed. Attempting to repair trailing commas.");
        try {
            const repaired = jsonCandidate.replace(/,(?=\s*[}\]])/g, '');
            JSON.parse(repaired);
            return repaired;
        } catch (repairError: any) {
            console.error(`[extractJson] CRITICAL FAILURE: Parsing failed even after repair.`, { 
                errorMessage: repairError.message,
                attemptedToParse: jsonCandidate
            });
            throw new Error(`Unable to parse JSON from AI response after multiple repair attempts.`);
        }
    }
};


/**
 * Extracts a YouTube video ID from various URL formats.
 * @param url The YouTube URL.
 * @returns The 11-character video ID or null if not found.
 */
const extractYouTubeID = (url: string): string | null => {
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*).*/;
    const match = url.match(regExp);
    if (match && match[2].length === 11) {
        return match[2];
    }
    return null;
};


/**
 * Extracts the final, clean slug from a URL, intelligently removing parent paths and file extensions.
 * This ensures a perfect match with the WordPress database slug.
 * @param urlString The full URL to parse.
 * @returns The extracted slug.
 */
const extractSlugFromUrl = (urlString: string): string => {
    try {
        const url = new URL(urlString);
        let pathname = url.pathname;

        // 1. Remove trailing slash to handle URLs like /my-post/
        if (pathname.endsWith('/') && pathname.length > 1) {
            pathname = pathname.slice(0, -1);
        }

        // 2. Get the last segment after the final '/'
        const lastSegment = pathname.substring(pathname.lastIndexOf('/') + 1);

        // 3. Remove common web file extensions like .html, .php, etc.
        const cleanedSlug = lastSegment.replace(/\.[a-zA-Z0-9]{2,5}$/, '');

        return cleanedSlug;
    } catch (error) {
        console.error("Could not parse URL to extract slug:", urlString, error);
        // Fallback for non-URL strings, though unlikely
        return urlString.split('/').pop() || '';
    }
};


/**
 * A more professional and resilient fetch function for AI APIs that includes
 * exponential backoff, respects 'Retry-After' headers, and intelligently fails fast on non-retriable errors.
 * @param apiCall A function that returns the promise from the AI SDK call.
 * @param maxRetries The maximum number of times to retry the call.
 * @param initialDelay The baseline delay in milliseconds for the first retry.
 * @returns The result of the successful API call.
 * @throws {Error} if the call fails after all retries or on a non-retriable error.
 */
const callAiWithRetry = async (apiCall: () => Promise<any>, maxRetries = 5, initialDelay = 5000) => {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            return await apiCall();
        } catch (error: any) {
            console.error(`AI call failed on attempt ${attempt + 1}. Error:`, error.message);

            const errorMessage = (error.message || '').toLowerCase();
            const statusMatch = errorMessage.match(/status(?: code)?: (\d{3})/);
            const statusCode = error.status || (statusMatch ? parseInt(statusMatch[1], 10) : null);

            // --- START: Stricter Non-Retriable Error Detection (Fail Fast) ---
            const isNonRetriableClientError = statusCode && statusCode >= 400 && statusCode < 500 && statusCode !== 429;
            const isContextLengthError = errorMessage.includes('context length') || errorMessage.includes('token limit') || errorMessage.includes('request too large');
            const isInvalidApiKeyError = errorMessage.includes('api key not valid');
            const isJsonGenerationError = errorMessage.includes('failed to generate json');

            if (isNonRetriableClientError || isContextLengthError || isInvalidApiKeyError || isJsonGenerationError) {
                console.error(`Encountered a non-retriable error (Status: ${statusCode}, Message: ${error.message}). Failing immediately.`);
                throw error;
            }
            // --- END: Stricter Non-Retriable Error Detection ---


            // If it's the last attempt for any other error, give up.
            if (attempt === maxRetries - 1) {
                console.error(`AI call failed on final attempt (${maxRetries}).`);
                throw error;
            }
            
            let delay: number = 0;
            
            // --- START: Enhanced 429 Rate Limit Handling ---
            if (statusCode === 429) {
                let delayFound = false;

                // 1. Check for 'Retry-After' header (standard)
                const retryAfterHeader = error.headers?.['retry-after'] || error.response?.headers?.get('retry-after');
                if (retryAfterHeader) {
                    const retryAfterSeconds = parseInt(retryAfterHeader, 10);
                    if (!isNaN(retryAfterSeconds)) {
                        delay = retryAfterSeconds * 1000 + 500; // Add 500ms buffer
                        console.log(`Rate limit hit. Provider requested a delay of ${retryAfterSeconds}s. Waiting...`);
                        delayFound = true;
                    } else {
                        const retryDate = new Date(retryAfterHeader);
                        if (!isNaN(retryDate.getTime())) {
                            delay = retryDate.getTime() - new Date().getTime() + 500;
                            console.log(`Rate limit hit. Provider requested waiting until ${retryDate.toISOString()}. Waiting...`);
                            delayFound = true;
                        }
                    }
                }

                // 2. If no header, check message body for "try again in X.Ys" (for APIs like Groq)
                if (!delayFound) {
                    const tryAgainMatch = errorMessage.match(/try again in ([\d\.]+)s/);
                    if (tryAgainMatch && tryAgainMatch[1]) {
                        const tryAgainSeconds = parseFloat(tryAgainMatch[1]);
                        delay = tryAgainSeconds * 1000 + 500; // Add 500ms buffer
                        console.log(`Rate limit hit. Message suggests retrying in ${tryAgainSeconds}s. Waiting...`);
                        delayFound = true;
                    }
                }
                
                // 3. If a delay was NOT found, decide if it's a hard quota error or needs backoff.
                if (!delayFound) {
                    const isHardQuotaError = errorMessage.includes('tokens per day') || errorMessage.includes('tpd') || errorMessage.includes('quota exceeded') || errorMessage.includes('daily limit');
                    if (isHardQuotaError) {
                        const dailyQuotaErrorMessage = `Daily quota exceeded for this model and no retry delay was provided. Please check your API plan or try again tomorrow.`;
                        console.error(dailyQuotaErrorMessage, { originalError: error.message });
                        throw new Error(dailyQuotaErrorMessage);
                    }
                    
                    // 4. Fallback to exponential backoff for generic 429s with no retry info.
                    delay = initialDelay * Math.pow(2, attempt) + (Math.random() * 1000);
                    console.log(`Rate limit hit. No 'Retry-After' or message delay found. Using exponential backoff.`);
                }
            } else {
                 // --- Standard Exponential Backoff for Server-Side Errors (5xx etc.) ---
                 const backoff = Math.pow(2, attempt);
                 const jitter = Math.random() * 1000;
                 delay = initialDelay * backoff + jitter;
            }
            // --- END: Enhanced 429 Rate Limit Handling ---

            console.log(`Retrying in ${Math.round(delay)}ms...`);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
    throw new Error("AI call failed after all retries.");
};

/**
 * Fetches a URL by first attempting a direct connection, then falling back to a
 * series of public CORS proxies. This strategy makes the sitemap crawling feature
 * significantly more resilient to CORS issues and unreliable proxies.
 * @param url The target URL to fetch.
 * @param options The options for the fetch call (method, headers, body).
 * @returns The successful Response object.
 * @throws {Error} if the direct connection and all proxies fail.
 */
const fetchWithProxies = async (url: string, options: RequestInit = {}): Promise<Response> => {
    let lastError: Error | null = null;
    const REQUEST_TIMEOUT = 20000; // 20 seconds

    // Standard headers to mimic a browser request, reducing the chance of being blocked.
    const browserHeaders = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9',
    };

    // --- NEW: Attempt a direct fetch first ---
    // This will work if the server has CORS enabled, and is the fastest option.
    try {
        console.log("Attempting direct fetch (no proxy)...");
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);
        const directResponse = await fetch(url, {
            ...options,
            headers: {
                ...browserHeaders,
                ...(options.headers || {}),
            },
            signal: controller.signal,
        });
        clearTimeout(timeoutId);
        if (directResponse.ok) {
            console.log("Successfully fetched directly (no proxy)!");
            return directResponse;
        }
    } catch (error: any) {
        // A TypeError here is the classic sign of a CORS error.
        if (error.name !== 'AbortError') { // Don't log timeout as a CORS error
            console.warn("Direct fetch failed (likely due to CORS). Proceeding with proxies.", error.name);
        }
        lastError = error;
    }

    // --- END: Direct fetch attempt ---

    const encodedUrl = encodeURIComponent(url);
    // An expanded and diversified list of public CORS proxies.
    const proxies = [
        `https://corsproxy.io/?${url}`,
        `https://api.codetabs.com/v1/proxy?quest=${encodedUrl}`,
        `https://api.allorigins.win/raw?url=${encodedUrl}`,
        `https://thingproxy.freeboard.io/fetch/${url}`,
    ];


    for (let i = 0; i < proxies.length; i++) {
        const proxyUrl = proxies[i];
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

        try {
            const shortProxyUrl = new URL(proxyUrl).hostname;
            console.log(`Attempting fetch via proxy #${i + 1} (${shortProxyUrl})...`);
            
            const response = await fetch(proxyUrl, {
                ...options,
                 headers: {
                    ...browserHeaders,
                    ...(options.headers || {}),
                },
                signal: controller.signal,
            });

            if (response.ok) {
                console.log(`Successfully fetched via proxy #${i + 1} (${shortProxyUrl})`);
                return response; // Success!
            }
            const responseText = await response.text().catch(() => `(could not read response body)`);
            lastError = new Error(`Proxy request failed with status ${response.status} for ${shortProxyUrl}. Response: ${responseText.substring(0, 100)}`);

        } catch (error: any) {
            if (error.name === 'AbortError') {
                const shortProxyUrl = new URL(proxyUrl).hostname;
                console.error(`Fetch via proxy #${i + 1} (${shortProxyUrl}) timed out after ${REQUEST_TIMEOUT / 1000}s.`);
                lastError = new Error(`Request timed out for proxy: ${shortProxyUrl}`);
            } else {
                console.error(`Fetch via proxy #${i + 1} failed:`, error);
                lastError = error as Error;
            }
        } finally {
            clearTimeout(timeoutId);
        }
    }

    // If we're here, all proxies failed.
    const baseErrorMessage = "Failed to crawl your sitemap. This is often due to a network or CORS issue where our proxy servers are blocked by your website's security (like Cloudflare or a firewall), or the target server is too slow to respond.\n\n" +
        "Please check that:\n" +
        "1. Your sitemap URL is correct and publicly accessible.\n" +
        "2. Your website's security settings aren't blocking anonymous proxy access.\n"
        "3. Your internet connection is stable.";

    throw new Error(lastError ? `${baseErrorMessage}\n\nLast Error: ${lastError.message}` : baseErrorMessage);
};


/**
 * Smartly fetches a WordPress API endpoint. If the request is authenticated, it forces a direct
 * connection, as proxies will strip authentication headers. Unauthenticated requests will use
 * the original proxy fallback logic.
 * @param targetUrl The full URL to the WordPress API endpoint.
 * @param options The options for the fetch call (method, headers, body).
 * @returns The successful Response object.
 * @throws {Error} if the connection fails.
 */
const fetchWordPressWithRetry = async (targetUrl: string, options: RequestInit): Promise<Response> => {
    const REQUEST_TIMEOUT = 30000; // 30 seconds for potentially large uploads
    const hasAuthHeader = options.headers && (options.headers as Record<string, string>)['Authorization'];

    // If the request has an Authorization header, it MUST be a direct request.
    // Proxies will strip authentication headers and cause a guaranteed failure.
    if (hasAuthHeader) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);
            const directResponse = await fetch(targetUrl, { ...options, signal: controller.signal });
            clearTimeout(timeoutId);
            return directResponse; // Return the response directly, regardless of status, to be handled by the caller.
        } catch (error: any) {
            if (error.name === 'AbortError') {
                throw new Error("WordPress API request timed out.");
            }
            // A TypeError is the classic sign of a CORS error on a failed fetch.
            // This will be caught and diagnosed by the calling function (e.g., verifyWpConnection)
            throw error;
        }
    }

    // --- Fallback to original proxy logic for NON-AUTHENTICATED requests ---
    let lastError: Error | null = null;
    
    // 1. Attempt Direct Connection
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);
        const directResponse = await fetch(targetUrl, { ...options, signal: controller.signal });
        clearTimeout(timeoutId);

        if (directResponse.ok || (directResponse.status >= 400 && directResponse.status < 500)) {
            return directResponse;
        }
        lastError = new Error(`Direct connection failed with status ${directResponse.status}`);
    } catch (error: any) {
        if (error.name !== 'AbortError') {
            console.warn("Direct WP API call failed (likely CORS or network issue). Trying proxies.", error.name);
        }
        lastError = error;
    }
    
    // 2. Attempt with Proxies if Direct Fails
    const encodedUrl = encodeURIComponent(targetUrl);
    const proxies = [
        `https://corsproxy.io/?${encodedUrl}`,
        `https://api.allorigins.win/raw?url=${encodedUrl}`,
    ];

    for (const proxyUrl of proxies) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);
        try {
            const shortProxyUrl = new URL(proxyUrl).hostname;
            console.log(`Attempting WP API call via proxy: ${shortProxyUrl}`);
            const response = await fetch(proxyUrl, { ...options, signal: controller.signal });
            if (response.ok || (response.status >= 400 && response.status < 500)) {
                console.log(`Successfully fetched via proxy: ${shortProxyUrl}`);
                return response;
            }
            const responseText = await response.text().catch(() => '(could not read response body)');
            lastError = new Error(`Proxy request failed with status ${response.status} for ${shortProxyUrl}. Response: ${responseText.substring(0, 100)}`);
        } catch (error: any) {
             if (error.name === 'AbortError') {
                const shortProxyUrl = new URL(proxyUrl).hostname;
                console.error(`Fetch via proxy ${shortProxyUrl} timed out.`);
                lastError = new Error(`Request timed out for proxy: ${shortProxyUrl}`);
            } else {
                lastError = error;
            }
        } finally {
            clearTimeout(timeoutId);
        }
    }

    throw lastError || new Error("All attempts to connect to the WordPress API failed.");
};


/**
 * Processes an array of items concurrently using async workers, with a cancellable mechanism.
 * @param items The array of items to process.
 * @param processor An async function that processes a single item.
 * @param concurrency The number of parallel workers.
 * @param onProgress An optional callback to track progress.
 * @param shouldStop An optional function that returns true to stop processing.
 */
async function processConcurrently<T>(
    items: T[],
    processor: (item: T) => Promise<void>,
    concurrency = 5,
    onProgress?: (completed: number, total: number) => void,
    shouldStop?: () => boolean
): Promise<void> {
    const queue = [...items];
    let completed = 0;
    const total = items.length;

    const run = async () => {
        while (queue.length > 0) {
            if (shouldStop?.()) {
                // Emptying the queue is a robust way to signal all workers to stop
                // after they finish their current task.
                queue.length = 0;
                break;
            }
            const item = queue.shift();
            if (item) {
                await processor(item);
                completed++;
                onProgress?.(completed, total);
            }
        }
    };

    const workers = Array(concurrency).fill(null).map(run);
    await Promise.all(workers);
};

/**
 * The "Anchor Text Guardian": A zero-tolerance function that scans AI-generated content
 * for the specific error of using a slug as anchor text and forcibly repairs it with the correct page title.
 * @param content The HTML content string with AI-generated placeholders.
 * @param availablePages An array of page objects from the sitemap, each with 'id', 'title', and 'slug'.
 * @returns The HTML content with incorrect anchor text placeholders repaired.
 */
const repairIncorrectAnchorText = (content: string, availablePages: any[]): string => {
    if (!content || !availablePages || availablePages.length === 0) {
        return content;
    }

    const pagesBySlug = new Map(availablePages.map(p => [p.slug, p]));
    const placeholderRegex = /\[INTERNAL_LINK\s+slug="([^"]+)"\s+text="([^"]+)"\]/g;

    return content.replace(placeholderRegex, (match, slug, text) => {
        // CRITICAL CHECK: If the anchor text is identical to the slug, it's an error.
        if (slug === text) {
            const page = pagesBySlug.get(slug);
            // Attempt to repair it using the page's title.
            if (page && page.title) {
                console.warn(`[Anchor Text Guardian] Found and FIXED incorrect anchor text for slug "${slug}". Replacing with title: "${page.title}"`);
                const sanitizedTitle = page.title.replace(/"/g, '&quot;');
                return `[INTERNAL_LINK slug="${slug}" text="${sanitizedTitle}"]`;
            }
        }
        // If the anchor text is not the slug, or we can't find a replacement title, return the original placeholder.
        return match;
    });
};


/**
 * Validates and repairs internal link placeholders from AI content. If an AI invents a slug,
 * this "Smart Link Forger" finds the best matching real page based on anchor text and repairs the link.
 * @param content The HTML content string with AI-generated placeholders.
 * @param availablePages An array of page objects from the sitemap, each with 'id', 'title', and 'slug'.
 * @returns The HTML content with invalid link placeholders repaired or removed.
 */
const validateAndRepairInternalLinks = (content: string, availablePages: any[]): string => {
    if (!content || !availablePages || availablePages.length === 0) {
        return content;
    }

    const pagesBySlug = new Map(availablePages.map(p => [p.slug, p]));
    const placeholderRegex = /\[INTERNAL_LINK\s+slug="([^"]+)"\s+text="([^"]+)"\]/g;

    return content.replace(placeholderRegex, (match, slug, text) => {
        // If the slug is valid and exists, we're good.
        if (pagesBySlug.has(slug)) {
            return match; // Return the original placeholder unchanged.
        }

        // --- Slug is INVALID. AI invented it. Time to repair. ---
        console.warn(`[Link Repair] AI invented slug "${slug}". Attempting to repair based on anchor text: "${text}".`);

        const anchorTextLower = text.toLowerCase();
        const anchorWords = anchorTextLower.split(/\s+/).filter(w => w.length > 2); // Meaningful words
        const anchorWordSet = new Set(anchorWords);
        let bestMatch: any = null;
        let highestScore = -1;

        for (const page of availablePages) {
            if (!page.slug || !page.title) continue;

            let currentScore = 0;
            const titleLower = page.title.toLowerCase();

            // Scoring Algorithm
            // 1. Exact title match (very high confidence)
            if (titleLower === anchorTextLower) {
                currentScore += 100;
            }
            
            // 2. Partial inclusion (high confidence)
            // - Anchor text is fully inside the title (e.g., anchor "SEO tips" in title "Advanced SEO Tips for 2025")
            if (titleLower.includes(anchorTextLower)) {
                currentScore += 60;
            }
            // - Title is fully inside the anchor text (rarer, but possible)
            if (anchorTextLower.includes(titleLower)) {
                currentScore += 50;
            }

            // 3. Keyword Overlap Score (the core of the enhancement)
            const titleWords = titleLower.split(/\s+/).filter(w => w.length > 2);
            if (titleWords.length === 0) continue; // Avoid division by zero
            
            const titleWordSet = new Set(titleWords);
            const intersection = new Set([...anchorWordSet].filter(word => titleWordSet.has(word)));
            
            if (intersection.size > 0) {
                // Calculate a relevance score based on how many words match
                const anchorMatchPercentage = (intersection.size / anchorWordSet.size) * 100;
                const titleMatchPercentage = (intersection.size / titleWordSet.size) * 100;
                // Average the two percentages. This rewards matches that are significant to both the anchor and the title.
                const overlapScore = (anchorMatchPercentage + titleMatchPercentage) / 2;
                currentScore += overlapScore;
            }

            if (currentScore > highestScore) {
                highestScore = currentScore;
                bestMatch = page;
            }
        }
        
        // Use a threshold to avoid bad matches
        if (bestMatch && highestScore > 50) {
            console.log(`[Link Repair] Found best match: "${bestMatch.slug}" with a score of ${highestScore.toFixed(2)}. Forging corrected link.`);
            const sanitizedText = text.replace(/"/g, '&quot;');
            return `[INTERNAL_LINK slug="${bestMatch.slug}" text="${sanitizedText}"]`;
        } else {
            console.warn(`[Link Repair] Could not find any suitable match for slug "${slug}" (best score: ${highestScore.toFixed(2)}). Removing link, keeping text.`);
            return text; // Fallback: If no good match, just return the anchor text.
        }
    });
};

// Helper function to escape characters for use in a regular expression
const escapeRegExp = (string: string): string => {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}

/**
 * The "Link Guardian 7.0": Re-engineered with a multi-tiered fallback system to guarantee
 * link injection even in poorly structured content, preventing critical failures and ensuring
 * the 6-12 internal link quota is always met inline.
 * @param content The HTML content, post-repair.
 * @param availablePages The sitemap page data.
 * @param primaryKeyword The primary keyword of the article being generated.
 * @param minLinks The minimum number of internal links required.
 * @param currentArticleSlug The slug of the article being processed to prevent self-linking.
 * @returns The HTML content with the link quota enforced inline.
 */
const enforceInternalLinkQuota = (content: string, availablePages: any[], primaryKeyword: string, minLinks: number, currentArticleSlug: string | null = null): string => {
    if (!availablePages || availablePages.length === 0) return content;

    const placeholderRegex = /\[INTERNAL_LINK\s+slug="([^"]+)"\s+text="([^"]+)"\]/g;
    
    // --- Natural Injection Phase ---
    const primaryKeywordWords = new Set(primaryKeyword.toLowerCase().split(/\s+/));
    
    const initialLinks = [...content.matchAll(placeholderRegex)];
    let linkedSlugs = new Set(initialLinks.map(match => match[1]));
    let linkedTexts = new Set(initialLinks.map(match => match[2].toLowerCase()));
    
    let deficit = minLinks - initialLinks.length;
    if (deficit <= 0) return content;

    const candidatePages = availablePages
        .filter(p => p.slug && !linkedSlugs.has(p.slug) && p.slug !== currentArticleSlug)
        .map(page => {
            if (!page.title) return { ...page, relevance: 0 };
            const titleWords = new Set(page.title.toLowerCase().split(/\s+/));
            const intersection = new Set([...primaryKeywordWords].filter(x => titleWords.has(x)));
            return { ...page, relevance: intersection.size };
        })
        .sort((a, b) => b.relevance - a.relevance);

    let newContent = content;

    for (const pageToLink of candidatePages) {
        if (deficit <= 0) break;
        if (!pageToLink.title) continue;
        
        let linkInjected = false;
        const phrase = pageToLink.title; // Use the full title as the primary search phrase
        const regex = new RegExp(`(?<=[>\\s.,(])(${escapeRegExp(phrase)})(?=[<\\s.,?!)]+)`, 'i');

        if (regex.test(newContent) && !linkedTexts.has(phrase.toLowerCase())) {
             newContent = newContent.replace(regex, (match) => {
                if (linkInjected) return match;
                const matchIndex = newContent.toLowerCase().indexOf(match.toLowerCase());
                const substringBefore = newContent.substring(Math.max(0, matchIndex - 50), matchIndex);
                if (/<a[^>]*>[^<]*$/i.test(substringBefore) || /\[INTERNAL_LINK[^\]]*\][^\[]*$/i.test(substringBefore)) return match;

                console.log(`[Link Guardian] Pass 1 (Natural): Injected link for "${pageToLink.slug}" using rich phrase: "${match}"`);
                linkedSlugs.add(pageToLink.slug);
                linkedTexts.add(match.toLowerCase());
                deficit--;
                linkInjected = true;
                return `[INTERNAL_LINK slug="${pageToLink.slug}" text="${match}"]`;
            });
        }
    }
    
    // --- Forced Injection Protocol (with Rich Anchor Text Purity Protocol) ---
    if (deficit > 0) {
        console.warn(`[Link Guardian] Natural linking insufficient. ${deficit} links short of quota. Activating Forced Injection Protocol.`);
        
        const isLikelyUrl = (str: string) => {
            if (!str) return true;
            const trimmed = str.trim();
            return trimmed.startsWith('http://') || trimmed.startsWith('https://') || /^[a-z0-9-]+\.[a-z]{2,}(\/.*)?$/i.test(trimmed);
        };

        const validCandidates = candidatePages.filter(p => {
            const isValid = p.slug && !linkedSlugs.has(p.slug) && p.title && p.title.trim().length >= 10 && !isLikelyUrl(p.title);
            if (!isValid) {
                console.warn(`[Link Guardian] Pre-filtering rejected candidate for injection: Slug="${p.slug}", Title="${p.title}"`);
            }
            return isValid;
        });

        if (validCandidates.length > 0) {
            let paragraphs = newContent.split('</p>');
            const potentialInjectionPoints: number[] = [];
            
            // Tier 1: Prefer longer paragraphs in the middle for better distribution.
            for (let i = 1; i < paragraphs.length - 1; i++) {
                if (paragraphs[i].replace(/<[^>]+>/g, '').trim().length > 150) {
                    potentialInjectionPoints.push(i);
                }
            }
            
            // Tier 2: If not enough spots, accept any non-trivial paragraph in the middle.
            const injectionPointSet = new Set(potentialInjectionPoints);
            if (injectionPointSet.size < deficit) {
                for (let i = 1; i < paragraphs.length - 1; i++) {
                    if (paragraphs[i].replace(/<[^>]+>/g, '').trim().length > 50) {
                        injectionPointSet.add(i);
                    }
                }
            }
            
            let finalInjectionPoints = Array.from(injectionPointSet);

            // Tier 3 (Failsafe): If still no points (e.g., very short article), use whatever we can.
            if (finalInjectionPoints.length === 0 && paragraphs.length > 1) {
                for (let i = 1; i < paragraphs.length; i++) {
                    if (paragraphs[i].replace(/<[^>]+>/g, '').trim().length > 10) {
                        finalInjectionPoints.push(i);
                    }
                }
            }

            // Shuffle for more natural-looking placement.
            finalInjectionPoints.sort(() => Math.random() - 0.5);

            if (finalInjectionPoints.length > 0) {
                let injectedCount = 0;
                for (let i = 0; i < Math.min(deficit, validCandidates.length); i++) {
                    const pageToLink = validCandidates[i];
                    
                    const injectionIndex = finalInjectionPoints[i % finalInjectionPoints.length];
                    
                    const anchorText = pageToLink.title!.replace(/"/g, '&quot;');
                    const linkPlaceholder = `[INTERNAL_LINK slug="${pageToLink.slug}" text="${anchorText}"]`;
                    
                    const sentences = [
                        ` For a deeper understanding, our guide on ${linkPlaceholder} is an excellent resource.`,
                        ` To learn more, consider reading about ${linkPlaceholder}.`,
                        ` We cover this in more detail in our article on ${linkPlaceholder}.`,
                        ` For more information on this topic, see our detailed guide on ${linkPlaceholder}.`
                    ];
                    const injectionSentence = sentences[i % sentences.length];
                    
                    paragraphs[injectionIndex] = paragraphs[injectionIndex] + injectionSentence;
                    
                    console.log(`[Link Guardian] Pass 4 (FORCE): Programmatically injected link for "${pageToLink.slug}" at paragraph index ${injectionIndex}.`);
                    injectedCount++;
                }
                newContent = paragraphs.join('</p>');
                deficit -= injectedCount;
            }
        }
    }

    if (deficit > 0) {
        console.error(`[Link Guardian] CRITICAL FAILURE: Could not meet link quota INLINE even with Forced Injection. ${deficit} links still missing. The content may be too short or lack valid pages to link to.`);
    }

    return newContent;
};


/**
 * Processes custom internal link placeholders in generated content and replaces them
 * with valid, full URL links based on a list of available pages.
 * @param content The HTML content string containing placeholders.
 * @param availablePages An array of page objects, each with 'id' (full URL) and 'slug'.
 * @returns The HTML content with placeholders replaced by valid <a> tags.
 */
const processInternalLinks = (content: string, availablePages: any[]): string => {
    if (!content || !availablePages || availablePages.length === 0) {
        return content;
    }

    // Create a map for efficient slug-to-page lookups.
    const pagesBySlug = new Map(availablePages.filter(p => p.slug).map(p => [p.slug, p]));

    // Regex to find placeholders like [INTERNAL_LINK slug="some-slug" text="some anchor text"]
    const placeholderRegex = /\[INTERNAL_LINK\s+slug="([^"]+)"\s+text="([^"]+)"\]/g;

    return content.replace(placeholderRegex, (match, slug, text) => {
        const page = pagesBySlug.get(slug);
        if (page && page.id) {
            // Found a valid page, create the link with the full URL.
            console.log(`[Link Processor] Found match for slug "${slug}". Replacing with link to ${page.id}`);
            // Escape quotes in text just in case AI includes them
            const sanitizedText = text.replace(/"/g, '&quot;');
            return `<a href="${page.id}">${sanitizedText}</a>`;
        } else {
            // This should rarely happen now with the new validation/repair/enforcement steps.
            console.warn(`[Link Processor] Could not find a matching page for slug "${slug}". This is unexpected. Replacing with plain text.`);
            return text; // Fallback: just return the anchor text.
        }
    });
};

const countSyllables = (word: string): number => {
    if (!word) return 0;
    word = word.toLowerCase().trim();
    if (word.length <= 3) { return 1; }
    word = word.replace(/(?:[^laeiouy]es|ed|[^laeiouy]e)$/, '');
    word = word.replace(/^y/, '');
    const matches = word.match(/[aeiouy]{1,2}/g);
    return matches ? matches.length : 0;
};

const calculateFleschReadability = (text: string): number => {
    const sentences = (text.match(/[.!?]+/g) || []).length || 1;
    const words = text.split(/\s+/).filter(Boolean).length;
    if (words < 100) return 0; // Not enough content for an accurate score

    let syllableCount = 0;
    text.split(/\s+/).forEach(word => {
        syllableCount += countSyllables(word);
    });

    const fleschScore = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllableCount / words);
    return Math.round(Math.min(100, Math.max(0, fleschScore)));
};

// --- END: Core Utility Functions ---


// --- TYPE DEFINITIONS ---
type GeneratedContent = {
    title: string;
    slug: string;
    metaDescription: string;
    primaryKeyword: string;
    semanticKeywords: string[];
    content: string;
    imageDetails: {
        prompt: string;
        altText: string;
        title: string;
        placeholder: string;
        generatedImageSrc?: string;
    }[];
    strategy: {
        targetAudience: string;
        searchIntent: string;
        competitorAnalysis: string;
        contentAngle: string;
    };
    jsonLdSchema: object;
    socialMediaCopy: {
        twitter: string;
        linkedIn: string;
    };
};

/**
 * Validates and normalizes the JSON object returned by the AI to ensure it
 * has all the required fields, preventing crashes from schema deviations.
 * @param parsedJson The raw parsed JSON from the AI.
 * @param itemTitle The original title of the content item, used for fallbacks.
 * @returns A new object with all required fields guaranteed to exist.
 */
const normalizeGeneratedContent = (parsedJson: any, itemTitle: string): GeneratedContent => {
    const normalized = { ...parsedJson };

    // --- Critical Fields ---
    if (!normalized.title) normalized.title = itemTitle;
    if (!normalized.slug) normalized.slug = itemTitle.toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]+/g, '');
    if (!normalized.content) {
        console.warn(`[Normalization] 'content' field was missing for "${itemTitle}". Defaulting to empty string.`);
        normalized.content = '';
    }

    // --- Image Details: The main source of errors ---
    if (!normalized.imageDetails || !Array.isArray(normalized.imageDetails) || normalized.imageDetails.length === 0) {
        console.warn(`[Normalization] 'imageDetails' was missing or invalid for "${itemTitle}". Generating default image prompts.`);
        const slugBase = normalized.slug || itemTitle.toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]+/g, '');
        normalized.imageDetails = [
            {
                prompt: `A high-quality, photorealistic image representing the concept of: "${normalized.title}". Cinematic, professional blog post header image, 16:9 aspect ratio.`,
                altText: `A conceptual image for "${normalized.title}"`,
                title: `${slugBase}-feature-image`,
                placeholder: '[IMAGE_1_PLACEHOLDER]'
            },
            {
                prompt: `An infographic or diagram illustrating a key point from the article: "${normalized.title}". Clean, modern design with clear labels. 16:9 aspect ratio.`,
                altText: `Infographic explaining a key concept from "${normalized.title}"`,
                title: `${slugBase}-infographic`,
                placeholder: '[IMAGE_2_PLACEHOLDER]'
            }
        ];
        
        // Ensure placeholders are injected if missing from content
        if (normalized.content && !normalized.content.includes('[IMAGE_1_PLACEHOLDER]')) {
            const paragraphs = normalized.content.split('</p>');
            if (paragraphs.length > 2) {
                paragraphs.splice(2, 0, '<p>[IMAGE_1_PLACEHOLDER]</p>');
                normalized.content = paragraphs.join('</p>');
            } else {
                normalized.content += '<p>[IMAGE_1_PLACEHOLDER]</p>';
            }
        }
        if (normalized.content && !normalized.content.includes('[IMAGE_2_PLACEHOLDER]')) {
            const paragraphs = normalized.content.split('</p>');
            if (paragraphs.length > 5) {
                paragraphs.splice(5, 0, '<p>[IMAGE_2_PLACEHOLDER]</p>');
                 normalized.content = paragraphs.join('</p>');
            } else {
                 normalized.content += '<p>[IMAGE_2_PLACEHOLDER]</p>';
            }
        }
    }

    // --- Other required fields for UI stability ---
    if (!normalized.metaDescription) normalized.metaDescription = `Read this comprehensive guide on ${normalized.title}.`;
    if (!normalized.primaryKeyword) normalized.primaryKeyword = itemTitle;
    if (!normalized.semanticKeywords || !Array.isArray(normalized.semanticKeywords)) normalized.semanticKeywords = [];
    if (!normalized.strategy) normalized.strategy = { targetAudience: '', searchIntent: '', competitorAnalysis: '', contentAngle: '' };
    if (!normalized.jsonLdSchema) normalized.jsonLdSchema = {};
    if (!normalized.socialMediaCopy) normalized.socialMediaCopy = { twitter: '', linkedIn: '' };

    return normalized as GeneratedContent;
};

const PROMPT_TEMPLATES = {
    cluster_planner: {
        systemInstruction: `You are a master SEO strategist specializing in building topical authority through pillar-and-cluster content models. Your task is to analyze a user's broad topic and generate a complete, SEO-optimized content plan that addresses user intent at every stage.

**RULES:**
1.  **Output Format:** Your entire response MUST be a single, valid JSON object. Do not include any text before or after the JSON.
2.  **Pillar Content:** The 'pillarTitle' must be a broad, comprehensive title for a definitive guide. It must be engaging, keyword-rich, and promise immense value to the reader. Think "The Ultimate Guide to..." or "Everything You Need to Know About...".
3.  **Cluster Content:** The 'clusterTitles' must be an array of 5 to 7 unique strings. Each title should be a compelling question or a long-tail keyword phrase that a real person would search for. These should be distinct sub-topics that logically support and link back to the main pillar page.
    - Good Example: "How Much Does Professional Landscaping Cost in 2025?"
    - Bad Example: "Landscaping Costs"
4.  **Keyword Focus:** All titles must be optimized for search engines without sounding robotic.
{{GEO_TARGET_INSTRUCTIONS}}
5.  **JSON Structure:** The JSON object must conform to this exact structure:
    {
      "pillarTitle": "A comprehensive, SEO-optimized title for the main pillar article.",
      "clusterTitles": [
        "A specific, long-tail keyword-focused title for the first cluster article.",
        "A specific, long-tail keyword-focused title for the second cluster article.",
        "..."
      ]
    }

**FINAL INSTRUCTION:** Your ENTIRE response MUST be ONLY the JSON object, starting with { and ending with }. Do not add any introductory text, closing remarks, or markdown code fences. Your output will be parsed directly by a machine.`,
        userPrompt: (topic: string) => `Generate a pillar-and-cluster content plan for the topic: "${topic}".`
    },
    premium_content_writer: {
        systemInstruction: () => `THE COMPLETE PRODUCTION-READY PREMIUM CONTENT ARCHITECTURE
***QUANTUM READABILITY & TRANSFORMATION PROTOCOL v5.0 INITIATED. ADHERENCE IS NON-NEGOTIABLE.***
***MISSION CRITICAL DIRECTIVE***
***OPERATING DATE: ${new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}***
***REAL-TIME DATA POWERED BY SERPER API***
***ENHANCED WITH MAXIMUM CLARITY & ENGAGEMENT ARCHITECTURE***

You are not an AI. You are a world-class human expert, a master educator, and a seasoned mentor in your field. Your SOLE function is to write the single greatest, most comprehensive, most engaging, and most profoundly helpful guide on the planet for the user's topic. Your writing must be simple, memorable, and create genuine transformation. You will empower the reader with true understanding, actionable strategies, and measurable results. You are writing for a smart audience, but you will explain everything with such clarity that an 8th grader could understand and apply it.

================================================================================
SECTION 1: CRITICAL PRE-GENERATION CHECKLIST & FAILURE CONDITIONS
================================================================================

Before generating a single token, you MUST mentally confirm your absolute commitment to the following. This is a non-negotiable pre-flight check. Failure to comply is a critical error.

1.  **JSON-ONLY OUTPUT:** I will output a single, PERFECTLY FORMED, minified, valid JSON object. Nothing else. No commentary, no markdown, NO EXCUSES.
2.  **8TH-GRADE READABILITY MANDATE:** My writing will be at an 8th-grade reading level. I will use short sentences (avg. < 15 words), simple words, and clear analogies. I will avoid jargon at all costs. My primary goal is CLARITY, not complexity.
3.  **ABSOLUTE MINIMUM WORD COUNT:** My final 'content' will ALWAYS exceed 2,000 words of high-value, insightful text. I will aim for 2,500+ words. No filler.
4.  **100% KEYWORD MANDATE:** I will use EVERY SINGLE semantic keyword provided. 100% of them. There are no exceptions. This is mission-critical.
5.  **PERFECT PAA INTEGRATION:** I will perfectly answer all "People Also Ask" questions both in the article body AND in the final FAQ section. This is a dual mandate.
6.  **MANDATORY "WOW" STATISTIC:** The article MUST begin with a verifiable, surprising statistic from 2025 that immediately hooks the reader.
7.  **2025+ DATA ONLY:** ALL statistics, trends, and data points MUST be from 2025 or later. Pre-2025 data = MISSION FAILURE.
8.  **FIRST-PERSON EXPERT VOICE:** I will write as THE expert with personal stories, specific experiences, and contrarian insights. I am a mentor, not an encyclopedia.
9.  **VISUAL DATA ARCHITECTURE:** I will include at least 5 tables/matrices for data visualization.
10. **STRATEGIC INTERNAL LINKING (6-12 Links):** THIS IS A NON-NEGOTIABLE, MISSION-CRITICAL REQUIREMENT. I will strategically embed EXACTLY 6 to 12 internal links using the provided <existing_articles_for_linking> list. Each link will use natural, contextually relevant, RICH ANCHOR TEXT (3+ words) and the format [INTERNAL_LINK slug="the-slug" text="the rich anchor text"]. I will ensure these links are embedded INLINE within the main body content, NEVER at the end or in a separate section.
11. **FEYNMAN TECHNIQUE FOR ALL CORE CONCEPTS:** I will explain every core concept as if I were teaching it to a smart 12-year-old. I will use simple language and relatable analogies.
12. **VIDEO EMBEDDING:** I WILL embed the two provided YouTube videos at the most contextually relevant points in the article. This is a mandatory requirement.
13. **REFERENCES SECTION PURITY:** The \`<h2>References</h2>\` section MUST contain ONLY the \`[REFERENCES_PLACEHOLDER]\` token and nothing else. NO other text, NO other links. This section is reserved for programmatic injection of external links ONLY.

**INSTANT FAILURE CONDITIONS:**
- Any output that isn't pure JSON = REJECTED
- Complex, jargon-filled writing = REJECTED
- Content under 2,000 words = REJECTED
- Missing even ONE semantic keyword = REJECTED
- Using pre-2025 data = REJECTED
- Generic, impersonal writing = REJECTED
- Fewer than 5 tables/visual elements = REJECTED
- Fewer than 6 or more than 12 internal links = REJECTED
- Using AI clichés or corporate speak = REJECTED

================================================================================
SECTION 2: ANTI-AI HUMANIZATION PROTOCOL 4.0 (Enhanced for Readability)
================================================================================

**FORBIDDEN PHRASES (NEVER USE):**
- "delve into", "navigate the landscape", "in the realm of", "unleash", "harness", "leverage"
- "elevate", "revolutionize", "seamless", "robust", "cutting-edge", "innovative"
- "in today's digital age", "the world of", "it's crucial to", "game-changer", "paradigm shift"
- "unlock the potential", "in conclusion", "transformative", "synergy", "streamline"
- "It's worth noting that", "Essentially", "Furthermore", "Moreover", "In the digital landscape"

**MANDATORY HUMAN & CLARITY ELEMENTS:**
- **Short Sentences & Paragraphs:** No paragraph will be longer than 4 sentences. Average sentence length will be under 15 words.
- **Simple Word Choice:** Use 'use' not 'utilize'. 'help' not 'facilitate'. 'show' not 'demonstrate'.
- **Active Voice Only:** "You should do this" not "This should be done."
- **Direct Address:** Speak directly to the reader ("You're probably wondering...").
- **Everyday Analogies:** "Think of it like baking a cake..." or "It's like compound interest for your skills."
- **Start sentences with "And", "But", "So,"** for conversational flow.
- **Use contractions naturally** (don't, it's, you're, I've).
- **Include parenthetical asides** (like this one) for a more personal tone.
- **"So What?" Test:** After every point, explain WHY it matters and WHAT the reader should do with it.

================================================================================
SECTION 3: ENHANCED JSON ARCHITECTURE WITH VISUAL ELEMENTS
================================================================================

{
  "title": "[Max 60 chars, contains primary keyword, emotionally compelling]",
  "slug": "[url-friendly-with-primary-keyword]",
  "metaDescription": "[Max 155 chars, action-oriented, contains primary keyword, creates urgency]",
  "primaryKeyword": "[exact primary keyword]",
  "semanticKeywords": ["every", "single", "keyword", "provided", "100%", "coverage"],
  "content": "[2500+ word HTML string following the ULTIMATE CONTENT BLUEPRINT. 8th-grade readability. ZERO FLUFF.]",
  "imageDetails": [
    {
      "prompt": "Photorealistic, professional image showing [specific scene], 16:9 aspect ratio, high detail, modern 2025 aesthetic, vibrant colors",
      "altText": "[Descriptive alt text with primary keyword]",
      "title": "[seo-friendly-filename]",
      "placeholder": "[IMAGE_1_PLACEHOLDER]"
    },
    {
      "prompt": "Simple, clean infographic illustrating [specific concept], 16:9 aspect ratio, 2025 design trends, clear labels, easy to understand at a glance",
      "altText": "[Descriptive alt text with semantic keyword]",
      "title": "[relevant-filename]",
      "placeholder": "[IMAGE_2_PLACEHOLDER]"
    },
    {
      "prompt": "Comparison chart or process flow showing [transformation], 16:9 aspect ratio, before/after visualization, minimalist design",
      "altText": "[Alt text with related keyword]",
      "title": "[descriptive-filename]",
      "placeholder": "[IMAGE_3_PLACEHOLDER]"
    }
  ],
  "strategy": {
    "targetAudience": "[Specific persona with pain points, goals, and current frustrations]",
    "searchIntent": "[informational/commercial/transactional/navigational]",
    "competitorAnalysis": "[Specific gaps identified and exactly how you filled them with simpler, more actionable advice]",
    "contentAngle": "[Unique 2025 perspective that simplifies a complex topic]",
    "internalLinkingStrategy": "[Strategic connection points to other pillars]",
    "promotionPlan": "[2025 omnichannel distribution strategy]",
    "engagementMetrics": "[Expected dwell time, scroll depth, interaction rate]"
  },
  "jsonLdSchema": {
    "@context": "https://schema.org",
    "@type": "Article",
    "headline": "[title]",
    "datePublished": "[current date]",
    "dateModified": "[current date]",
    "author": {"@type": "Person", "name": "[Expert Author Name]", "url": "[author bio URL]"},
    "publisher": {
      "@type": "Organization",
      "name": "[Publisher]",
      "logo": {"@type": "ImageObject", "url": "[logo URL]"}
    },
    "description": "[meta description]",
    "mainEntityOfPage": {"@type": "WebPage", "@id": "[full URL]"},
    "wordCount": "[actual word count]",
    "articleSection": "[category]",
    "keywords": "[comma-separated keywords]"
  },
  "socialMediaCopy": {
    "twitter": "[280 char hook with shocking 2025 statistic + question]",
    "linkedIn": "[Professional angle with current data and thought leadership hook]",
    "facebook": "[Emotional angle with curiosity gap]",
    "email": "[Subject line + preview text for newsletter]"
  }
}

================================================================================
SECTION 4: ULTIMATE CONTENT CONSTRUCTION BLUEPRINT (2500+ WORDS, 8TH-GRADE READABILITY)
================================================================================

**THE 'CONTENT' FIELD MUST FOLLOW THIS EXACT ENHANCED STRUCTURE:**

### 1. THE QUANTUM HOOK (200 words)
<h1>[Emotionally Charged Title with Primary Keyword]</h1>

<p><strong>In 2025, a stunning [X]% of [people/businesses] still get this wrong.</strong> Let that sink in. [Connect this statistic to their deepest fear or desire in a simple sentence].</p>

<p>Look, [agitate the core problem with another shocking 2025 data point]. But here's the good news: there's a much simpler way to [achieve desired outcome]. [Bold promise of a specific, tangible transformation with a timeframe].</p>

<div class="quick-win">
<strong>⚡ 30-Second Win:</strong> [Ultra-specific action they can take right now that proves your method is easy]
</div>

### 2. EXECUTIVE SUMMARY: YOUR PATH TO MASTERY (200 words)
<div class="key-takeaways">
<h2>Here’s What You'll Master in 12 Minutes</h2>
<ul>
<li><strong>First 3 Mins:</strong> Why [conventional wisdom] is a trap (and the simple truth they miss).</li>
<li><strong>Next 3 Mins:</strong> The [Unique Framework Name] – a 3-step system for [specific result].</li>
<li><strong>Next 3 Mins:</strong> My copy-paste templates for [desired outcome].</li>
<li><strong>Final 3 Mins:</strong> The #1 mistake that costs people [specific loss], and how to avoid it forever.</li>
</ul>
<div class="value-promise">
<strong>Bottom Line:</strong> This guide gives you a proven roadmap. Follow it, and you'll achieve [specific transformation] by this time next month.
</div>
</div>

### 3. THE STORY: MY $[SPECIFIC NUMBER] MISTAKE (500 words)
<h2>How a Single Mistake Cost Me $[Number] (And Taught Me Everything)</h2>

<p>It was a Tuesday morning. I hit 'refresh' and my heart sank. [Describe the specific failure metric]. All that work, down the drain.</p>
<p>[Vivid, simple description of the failure moment - what you saw, felt].</p>
<p>But that failure forced me to uncover a simple truth most people overlook. [Explain the 'aha' moment in simple terms]. It wasn't about working harder. It was about [the core insight].</p>
<p>Fast forward to today: [specific current success metric]. Same market, same me. But a completely different, simpler approach.</p>
<div class="pro-tip">
<strong>🎯 Key Insight:</strong> [Distilled wisdom in one simple, memorable sentence].
</div>

### 4. THE FOUNDATION: GETTING THE BASICS RIGHT (400 words)
<h2>The 2025 Rules: What's Changed and Why It Matters</h2>

<p>Okay, let's break this down. Think of [your topic] like [simple, everyday analogy]. Most people focus on [the complicated part], but the real secret is in [the simple part].</p>
<p>Here’s why this is so important now: [Explain the fundamental concept using the Feynman Technique. Use 2025 market data as proof].</p>
[Answer 1-2 PAA questions naturally within this section]

<div class="comparison-matrix">
<h3>Old Way vs. New Way (2025)</h3>
<table>
<thead>
<tr><th>Approach</th><th>Effort</th><th>Cost</th><th>Result</th><th>Who it's for</th></tr>
</thead>
<tbody>
<tr><td>The Old, Complex Way</td><td>High</td><td>High</td><td>Slow, Unpredictable</td><td>People who like headaches</td></tr>
<tr><td>The New, Simple Way</td><td>Low</td><td>Low</td><td>Fast, Consistent</td><td>People who want results</td></tr>
</tbody>
</table>
</div>

### 5. THE CORE SYSTEM: YOUR STEP-BY-STEP BLUEPRINT (1200 words)
<h2>The [Unique Branded Framework Name]: A 3-Step Plan for [Result]</h2>

<div class="methodology-overview">
<p>This is the exact system I use. It has 3 simple steps. Let's walk through them.</p>
</div>

<h3>Step 1: [Simple, Action-Oriented Name]</h3>

<p>This is where most people get tripped up. They think they need to [common mistake]. But you only need to do one thing: [the simple, correct action].</p>
<p><strong>So what?</strong> This saves you [X hours] and ensures you avoid [common problem].</p>

<div class="implementation-checklist">
<h4>Your Step 1 Checklist:</h4>
<ul>
<li>☐ [Specific action with a clear outcome]</li>
<li>☐ [Tool to use (mention a free one if possible)]</li>
<li>☐ [Metric to check with a target number]</li>
</ul>
</div>
[IMAGE_1_PLACEHOLDER]

<h3>Step 2: [Simple, Action-Oriented Name]</h3>

<p>Now that you've got your foundation, it's time to accelerate. This part is surprisingly easy. All you do is [the simple action].</p>
<p><strong>Think of it like this:</strong> [Another simple analogy].</p>

<div class="before-after-transformation">
<h4>Your Progress: Before vs. After Step 2</h4>
<table>
<tr><th>Area</th><th>Starting Point</th><th>Your New Result</th><th>Improvement</th></tr>
<tr><td>[Metric 1]</td><td>[Starting value]</td><td>[End value]</td><td>[X%]</td></tr>
<tr><td>[Metric 2]</td><td>[Starting value]</td><td>[End value]</td><td>[X%]</td></tr>
</table>
</div>

<h3>Step 3: [Simple, Action-Oriented Name]</h3>

<p>This is the final step to lock in your results. It's about consistency. [Explain the final simple action].</p>
<p><strong>Here's the secret:</strong> You don't need to be perfect. You just need to be consistent for [X days].</p>

<div class="roi-calculator">
<h4>The Payoff: Why This Is Worth It</h4>
<table>
<tr><th>Investment</th><th>Time</th><th>Expected Return</th><th>ROI</th></tr>
<tr><td>Following this system</td><td>30 mins/day</td><td>[Specific positive outcome]</td><td>Massive</td></tr>
</table>
</div>
[IMAGE_2_PLACEHOLDER]

### 6. THE BIGGEST MYTHS (DEBUNKED) (300 words)
<h2>3 Dangerous Myths That Are Holding You Back</h2>

<div class="myth-buster-table">
<table>
<tr><th>The Myth</th><th>The Simple Truth (2025 Data)</th><th>What to Do Instead</th></tr>
<tr><td>"[Common myth 1]"</td><td>[Reality, backed by data]</td><td>[A simple, 1-sentence action]</td></tr>
<tr><td>"[Common myth 2]"</td><td>[Reality, backed by data]</td><td>[A simple, 1-sentence action]</td></tr>
<tr><td>"[Common myth 3]"</td><td>[Reality, backed by data]</td><td>[A simple, 1-sentence action]</td></tr>
</table>
</div>

### 7. YOUR ACTION PLAN: THE NEXT 30 DAYS (500 words)
<h2>Your Day-by-Day Action Plan</h2>
<p>Don't just read this. Do it. Here is your plan for the next 4 weeks.</p>
<div class="implementation-roadmap">
<h3>Week 1: Build Your Foundation</h3>
<table>
<tr><th>Day</th><th>Your 30-Minute Task</th><th>Goal</th></tr>
<tr><td>1-3</td><td>[Specific simple task]</td><td>[Measurable outcome]</td></tr>
<tr><td>4-7</td><td>[Specific simple task]</td><td>[Measurable outcome]</td></tr>
</table>
<h3>Weeks 2-4: Build Momentum</h3>
[Similar simple, clear structure]
</div>
[IMAGE_3_PLACEHOLDER]

### 8. FREQUENTLY ASKED QUESTIONS (MANDATORY - All PAA)
<h2>Your Questions, Answered</h2>

<h3>[First PAA Question]?</h3>
<p>Good question. The simple answer is [provide a direct, simple answer]. Here's what that means for you: [explain the benefit in one sentence].</p>

[Answer ALL PAA questions provided in the same simple, direct format]

### 9. YOUR NEXT STEP (200 words)
<h2>What to Do Right Now</h2>

<div class="immediate-actions">
<p>You have a choice. You can close this tab and change nothing. Or you can take 2 minutes and start right now.</p>
<ol>
<li>
<strong>First (2 minutes):</strong> [Ultra-specific first action]. Go do it. I'll wait.
</li>
<li>
<strong>Next (Tonight):</strong> [Second action that builds momentum].
</li>
</ol>
</div>
<div class="final-motivation">
<h3>The Bottom Line:</h3>
<p>I've given you the simplest, most effective plan that exists. The only thing left is for you to follow it.</p>
</div>

### 10. RESOURCES & TOOLS (MANDATORY)
<h2>References</h2>
<div class="references-section">
[REFERENCES_PLACEHOLDER]
</div>
<h3>My Recommended Tools (2025)</h3>
<div class="tools-table">
<table>
<tr><th>Tool</th><th>Purpose</th><th>Free Alternative?</th></tr>
<tr><td>[Tool 1]</td><td>[Specific use]</td><td>[Yes/No]</td></tr>
[2-3 more tools]
</table>
</div>

================================================================================
FINAL COMMAND
================================================================================

BEGIN JSON OUTPUT IMMEDIATELY. NO COMMENTARY. NO MARKDOWN. PURE, SIMPLE, TRANSFORMATIONAL EXCELLENCE.`,
        userPrompt: (primaryKeyword: string, semanticKeywords: string[] | null, serpData: any[] | null, paaQuestions: string[] | null, relatedSearches: string[] | null, existingPages: any[] | null, youtubeVideos: any[] | null, originalContent: string | null = null, currentSlug: string | null = null) => {
            const MAX_CONTENT_CHARS = 5000; // Further reduced for safety
            const MAX_LINKING_PAGES = 30; 
            const MAX_SERP_SNIPPET_LENGTH = 100; // Further reduced for safety
            const MAX_RELATED_SEARCHES = 8;

            let contentForPrompt = '';
            if (originalContent) {
                const truncatedContent = originalContent.length > MAX_CONTENT_CHARS 
                    ? originalContent.substring(0, MAX_CONTENT_CHARS) 
                    : originalContent;
                contentForPrompt = `
***CRITICAL REWRITE MANDATE:*** You are to deconstruct the following outdated article and rebuild it into the ultimate guide, adhering strictly to all protocols.
<original_content_to_rewrite>
${truncatedContent}
</original_content_to_rewrite>
`;
            }

            // --- STATE-OF-THE-ART: Intelligent Link Curation Engine ---
            let relevantArticlesForLinking = [];
            if (existingPages && existingPages.length > 0) {
                const primaryKeywordLower = primaryKeyword.toLowerCase();
                const primaryKeywordWords = new Set(primaryKeywordLower.split(/\s+/).filter(w => w.length > 2));

                const scoredPages = existingPages
                    .map(page => {
                        if (!page.title) return { ...page, relevanceScore: -1 };
                        
                        const titleLower = page.title.toLowerCase();
                        let score = 0;

                        // Highest score for exact match
                        if (titleLower === primaryKeywordLower) {
                            score = 1000;
                        } else {
                            const titleWords = titleLower.split(/\s+/);
                            const titleWordSet = new Set(titleWords.filter(w => w.length > 2));
                            
                            // Score based on number of overlapping keywords
                            const intersection = new Set([...primaryKeywordWords].filter(word => titleWordSet.has(word)));
                            score += intersection.size * 10;

                            // Bonus if the primary keyword is a substring of the title
                            if (titleLower.includes(primaryKeywordLower)) {
                                score += 50;
                            }
                        }
                        
                        return { ...page, relevanceScore: score };
                    })
                    .filter(page => page.relevanceScore > 0 && (!currentSlug || page.slug !== currentSlug))
                    .sort((a, b) => b.relevanceScore - a.relevanceScore);
                
                relevantArticlesForLinking = scoredPages.slice(0, MAX_LINKING_PAGES);
                console.log(`[Link Curation] Selected top ${relevantArticlesForLinking.length} relevant pages for keyword "${primaryKeyword}".`, relevantArticlesForLinking.map(p => ({ title: p.title, score: p.relevanceScore })));
            }
            // --- END: Intelligent Link Curation Engine ---

            return `
================================================================================
SECTION 10: DYNAMIC PROMPT INTEGRATION
================================================================================

Generate an expert-level, comprehensive, transformational pillar article for the primary keyword: "${primaryKeyword}".
${contentForPrompt}

${semanticKeywords ? `

**SEMANTIC KEYWORDS (100% MANDATORY):** You MUST integrate every single keyword from this list naturally throughout the content:
<semantic_keywords>${JSON.stringify(semanticKeywords)}</semantic_keywords>` : ''}

${serpData ? `

**CRITICAL SERP COMPETITOR DATA:** Real-time data from top 10 competitors (via Serper API). Your reference links MUST come from this data.
<serp_data>${JSON.stringify(serpData.map(d => ({title: d.title, link: d.link, snippet: d.snippet?.substring(0, MAX_SERP_SNIPPET_LENGTH)})))}</serp_data>` : ''}

${paaQuestions ? `

**PEOPLE ALSO ASK (DUAL PLACEMENT MANDATORY):** Answer these thoroughly in the article body AND summarize in FAQ section:
<paa_questions>${JSON.stringify(paaQuestions)}</paa_questions>` : ''}

${relatedSearches ? `

**RELATED SEARCHES (TOPICAL AUTHORITY):** Cover these concepts for comprehensive topical coverage:
<related_searches>${JSON.stringify(relatedSearches.slice(0, MAX_RELATED_SEARCHES))}</related_searches>` : ''}

${relevantArticlesForLinking.length > 0 ? `

**INTERNAL LINKING TARGETS:** Link to these existing articles where contextually relevant:
<existing_articles_for_linking>${JSON.stringify(relevantArticlesForLinking.map(p => ({slug: p.slug, title: p.title})).filter(p => p.slug && p.title))}</existing_articles_for_linking>` : ''}

${youtubeVideos && youtubeVideos.length > 0 ? `

**VIDEO EMBEDS (MANDATORY):** Embed these videos at optimal points with proper H3 headings:
<div class="video-container"><iframe src="${youtubeVideos[0]?.embedUrl}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen title="${youtubeVideos[0]?.title}"></iframe></div>

<youtube_videos>${JSON.stringify(youtubeVideos)}</youtube_videos>` : ''}

**CONTEXT & REQUIREMENTS:**
- Current Date: ${new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
- Current Quarter: Q${Math.ceil((new Date().getMonth() + 1) / 3)} 2025
- Minimum Word Count: 2,500 (no maximum)
- Readability: 8th Grade Level
- Data Requirement: 2025 or later ONLY
- Voice: First-person expert mentor
- Style: Simple, clear, conversational, transformational
- Output: Single minified JSON object

Execute your expert analysis protocol. Create the definitive resource that transforms readers' lives.`;
        }
    },
    semantic_keyword_generator: {
        systemInstruction: `You are a world-class SEO analyst. Your task is to generate a comprehensive list of semantic and LSI (Latent Semantic Indexing) keywords related to a primary topic. These keywords should cover sub-topics, user intent variations, and related entities.

**RULES:**
1.  **Output Format:** Your entire response MUST be a single, valid JSON object. Do not include any text, markdown, or justification before or after the JSON.
2.  **Quantity:** Generate between 15 and 25 keywords.
3.  **JSON Structure:** The JSON object must conform to this exact structure:
    {
      "semanticKeywords": [
        "A highly relevant LSI keyword.",
        "A long-tail question-based keyword.",
        "Another related keyword or phrase.",
        "..."
      ]
    }

**FINAL INSTRUCTION:** Your ENTIRE response MUST be ONLY the JSON object, starting with { and ending with }. Do not add any introductory text, closing remarks, or markdown code fences. Your output will be parsed directly by a machine.`,
        userPrompt: (primaryKeyword: string) => `Generate semantic keywords for the primary topic: "${primaryKeyword}".`
    },
    content_health_analyzer: {
        systemInstruction: `You are an expert SEO content auditor. Your task is to analyze the provided text from a blog post and assign it a "Health Score". A low score indicates the content is thin, outdated, poorly structured, or not helpful, signaling an urgent need for an update.

**Evaluation Criteria:**
*   **Content Depth & Helpfulness (40%):** How thorough is the content? Does it seem to satisfy user intent? Is it just surface-level, or does it provide real value?
*   **Readability & Structure (30%):** Is it well-structured with clear headings? Are paragraphs short and scannable? Is the language complex or easy to read?
*   **Engagement Potential (15%):** Does it use lists, bullet points, or other elements that keep a reader engaged?
*   **Freshness Signals (15%):** Does the content feel current, or does it reference outdated concepts, statistics, or years?

**RULES:**
1.  **Output Format:** Your entire response MUST be a single, valid JSON object. Do not include any text, markdown, or justification before or after the JSON.
2.  **Health Score:** The 'healthScore' must be an integer between 0 and 100.
3.  **Update Priority:** The 'updatePriority' must be one of: "Critical" (score 0-25), "High" (score 26-50), "Medium" (score 51-75), or "Healthy" (score 76-100).
4.  **Justification:** Provide a concise, one-sentence explanation for your scoring in the 'justification' field.
5.  **JSON Structure:**
    {
      "healthScore": 42,
      "updatePriority": "High",
      "justification": "The content covers the topic superficially and lacks clear structure, making it difficult to read."
    }

**FINAL INSTRUCTION:** Your ENTIRE response MUST be ONLY the JSON object, starting with { and ending with }. Do not add any introductory text, closing remarks, or markdown code fences. Your output will be parsed directly by a machine.`,
        userPrompt: (content: string) => `Analyze the following blog post content and provide its SEO health score.\n\n&lt;content&gt;\n${content}\n&lt;/content&gt;`
    }
};

type ContentItem = {
    id: string;
    title: string;
    type: 'pillar' | 'cluster' | 'standard';
    status: 'idle' | 'generating' | 'done' | 'error';
    statusText: string;
    generatedContent: GeneratedContent | null;
    crawledContent: string | null;
    originalUrl?: string;
};

type SeoCheck = {
    valid: boolean;
    text: string;
    count?: number;
};


// --- REDUCER for items state ---
type ItemsAction =
    | { type: 'SET_ITEMS'; payload: Partial<ContentItem>[] }
    | { type: 'UPDATE_STATUS'; payload: { id: string; status: ContentItem['status']; statusText: string } }
    | { type: 'SET_CONTENT'; payload: { id: string; content: GeneratedContent } }
    | { type: 'SET_CRAWLED_CONTENT'; payload: { id: string; content: string } };

const itemsReducer = (state: ContentItem[], action: ItemsAction): ContentItem[] => {
    switch (action.type) {
        case 'SET_ITEMS':
            return action.payload.map((item: any) => ({ ...item, status: 'idle', statusText: 'Not Started', generatedContent: null, crawledContent: item.crawledContent || null }));
        case 'UPDATE_STATUS':
            return state.map(item =>
                item.id === action.payload.id
                    ? { ...item, status: action.payload.status, statusText: action.payload.statusText }
                    : item
            );
        case 'SET_CONTENT':
            return state.map(item =>
                item.id === action.payload.id
                    ? { ...item, status: 'done', statusText: 'Completed', generatedContent: action.payload.content }
                    : item
            );
        case 'SET_CRAWLED_CONTENT':
             return state.map(item =>
                item.id === action.payload.id
                    ? { ...item, crawledContent: action.payload.content }
                    : item
            );
        default:
            return state;
    }
};

// --- Child Components ---

const ProgressBar = memo(({ currentStep, onStepClick }: { currentStep: number; onStepClick: (step: number) => void; }) => {
    const steps = ["Setup", "Content Strategy", "Review & Export"];
    return (
        <nav aria-label="Main navigation">
            <ol className="progress-bar">
                {steps.map((name, index) => {
                    const stepIndex = index + 1;
                    const isClickable = true; // All steps are clickable
                    const status = stepIndex < currentStep ? 'completed' : stepIndex === currentStep ? 'active' : 'upcoming';
                    return (
                        <li 
                            key={index} 
                            className={`progress-step ${status} ${isClickable ? 'clickable' : ''}`} 
                            aria-current={status === 'active'}
                            onClick={() => isClickable && onStepClick(stepIndex)}
                            role="button"
                            tabIndex={0}
                        >
                            <div className="step-circle">{status === 'completed' ? '✓' : stepIndex}</div>
                            <span className="step-name">{name}</span>
                        </li>
                    );
                })}
            </ol>
        </nav>
    );
});


interface ApiKeyInputProps {
    provider: string;
    value: string;
    onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => void;
    status: 'idle' | 'validating' | 'valid' | 'invalid';
    name?: string;
    placeholder?: string;
    isTextArea?: boolean;
    isEditing: boolean;
    onEdit: () => void;
    type?: 'text' | 'password';
}
const ApiKeyInput = memo(({ provider, value, onChange, status, name, placeholder, isTextArea, isEditing, onEdit, type = 'password' }: ApiKeyInputProps) => {
    const InputComponent = isTextArea ? 'textarea' : 'input';

    if (status === 'valid' && !isEditing) {
        return (
            <div className="api-key-group">
                <input type="text" readOnly value={`**** **** **** ${value.slice(-4)}`} />
                <button onClick={onEdit} className="btn-edit-key" aria-label={`Edit ${provider} API Key`}>Edit</button>
            </div>
        );
    }

    const commonProps = {
        name: name || `${provider}ApiKey`,
        value: value,
        onChange: onChange,
        placeholder: placeholder || `Enter your ${provider.charAt(0).toUpperCase() + provider.slice(1)} API Key`,
        'aria-invalid': status === 'invalid',
        'aria-describedby': `${provider}-status`,
        ...(isTextArea ? { rows: 4 } : { type: type })
    };

    return (
        <div className="api-key-group">
            <InputComponent {...commonProps} />
            <div className="key-status-icon" id={`${provider}-status`} role="status">
                {status === 'validating' && <div className="key-status-spinner" aria-label="Validating key"></div>}
                {status === 'valid' && <svg className="success" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-label="Key is valid"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" /></svg>}
                {status === 'invalid' && <svg className="error" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-label="Key is invalid"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" /></svg>}
            </div>
        </div>
    );
});

const SeoChecklist = memo(({ checks }: { checks: Record<string, SeoCheck> }) => (
    <ul className="guardian-checklist">
        {Object.entries(checks).map(([key, check]) => (
            <li key={key} className={check.valid ? 'valid' : 'invalid'}>
                {check.valid ? (
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-label="Success"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" /></svg>
                ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-label="Error"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" /></svg>
                )}
                <span>{check.text}</span>
            </li>
        ))}
    </ul>
));

interface RankGuardianProps {
    seoData: {
        title: string;
        metaDescription: string;
        slug: string;
        primaryKeyword: string;
        content: string;
    };
}
const RankGuardian = memo(({ seoData }: RankGuardianProps) => {
    const checks = useMemo(() => {
        const { title, metaDescription, primaryKeyword, content } = seoData;
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = content || '';
        const textContent = tempDiv.textContent || '';
        const wordCount = textContent.split(/\s+/).filter(Boolean).length;
        const h1s = tempDiv.getElementsByTagName('h1').length;
        const h2s = tempDiv.getElementsByTagName('h2').length;

        const titleChecks: Record<string, SeoCheck> = {
            length: { valid: title.length > 0 && title.length <= 60, text: `Title is ${title.length} chars (1-60 ideal)` },
            keyword: { valid: title.toLowerCase().includes(primaryKeyword.toLowerCase()), text: 'Title contains keyword' }
        };

        const metaChecks: Record<string, SeoCheck> = {
            length: { valid: metaDescription.length > 0 && metaDescription.length <= 155, text: `Meta is ${metaDescription.length} chars (1-155 ideal)` },
            keyword: { valid: metaDescription.toLowerCase().includes(primaryKeyword.toLowerCase()), text: 'Meta contains keyword' }
        };
        
        const contentChecks: Record<string, SeoCheck> = {
            wordCount: { valid: wordCount >= 300, text: `${wordCount} words (300+ recommended)` },
            keywordDensity: {
                count: (textContent.toLowerCase().match(new RegExp(primaryKeyword.toLowerCase(), 'g')) || []).length,
                get valid() { return this.count! > 0; },
                get text() { return `Keyword used ${this.count} time(s)`; }
            },
            h1: { valid: h1s === 0, text: `${h1s} H1 tags (0 is required in content)` },
            h2: { valid: h2s >= 2, text: `${h2s} H2 tags (2+ recommended)` }
        };
        
        const readabilityScore = calculateFleschReadability(textContent);
        const allChecks = { ...titleChecks, ...metaChecks, ...contentChecks };
        const seoScore = Object.values(allChecks).filter(c => c.valid).length;
        const totalChecks = Object.keys(allChecks).length;
        const seoScorePercent = Math.round((seoScore / totalChecks) * 100);

        return { titleChecks, metaChecks, contentChecks, readabilityScore, seoScorePercent };
    }, [seoData]);

    const ScoreCircle = ({ score, label }: { score: number; label: string }) => {
        const radius = 40;
        const circumference = 2 * Math.PI * radius;
        const offset = circumference - (score / 100) * circumference;
        let strokeColor = 'var(--success-color)';
        if (score < 75) strokeColor = 'var(--warning-text-color)';
        if (score < 50) strokeColor = 'var(--error-color)';

        return (
            <div className="score-circle-container" role="meter" aria-valuenow={score} aria-valuemin={0} aria-valuemax={100} aria-label={`${label} score is ${score} out of 100`}>
                <svg className="score-circle" viewBox="0 0 100 100">
                    <circle className="circle-bg" cx="50" cy="50" r={radius}></circle>
                    <circle
                        className="circle"
                        cx="50"
                        cy="50"
                        r={radius}
                        stroke={strokeColor}
                        strokeDasharray={circumference}
                        strokeDashoffset={offset}
                    ></circle>
                </svg>
                <span className="score-text" style={{ color: strokeColor }}>{score}</span>
            </div>
        );
    };

    return (
        <div className="rank-guardian-pane">
            <div className="score-display">
                <div className="score-card">
                    <h4>SEO Score</h4>
                    <ScoreCircle score={checks.seoScorePercent} label="SEO"/>
                </div>
                <div className="score-card">
                    <h4>Readability</h4>
                    <ScoreCircle score={checks.readabilityScore} label="Readability" />
                </div>
            </div>
            <div className="checklists-container">
                <div className="checklist-column">
                    <h5>Title & Meta</h5>
                    <SeoChecklist checks={{ ...checks.titleChecks, ...checks.metaChecks }} />
                </div>
                <div className="checklist-column">
                    <h5>Content</h5>
                    <SeoChecklist checks={checks.contentChecks} />
                </div>
            </div>
        </div>
    );
});

interface ReviewModalProps {
    item: ContentItem;
    onClose: () => void;
    onSaveChanges: (itemId: string, updatedSeo: { title: string; metaDescription: string; slug: string }, updatedContent: string) => void;
    wpConfig: { url: string, username: string };
    wpPassword: string;
    onPublishSuccess: (originalUrl: string) => void;
    publishItem: (itemToPublish: ContentItem, currentWpPassword: string) => Promise<{ success: boolean; message: React.ReactNode; link?: string }>;
}

const ReviewModal = ({ item, onClose, onSaveChanges, wpConfig, wpPassword, onPublishSuccess, publishItem }: ReviewModalProps) => {
    if (!item || !item.generatedContent) return null;

    const [activeTab, setActiveTab] = useState('Live Preview');
    const [activeSeoTab, setActiveSeoTab] = useState('serp');
    const [editedSeo, setEditedSeo] = useState({ title: '', metaDescription: '', slug: '' });
    const [editedContent, setEditedContent] = useState('');
    const [copyStatus, setCopyStatus] = useState('Copy HTML');
    const [wpPublishStatus, setWpPublishStatus] = useState('idle'); // idle, publishing, success, error
    const [wpPublishMessage, setWpPublishMessage] = useState<React.ReactNode>('');


    useEffect(() => {
        if (item && item.generatedContent) {
            const isUpdate = !!item.originalUrl;
            // CRITICAL FIX: If this is an update, extract and enforce the original slug.
            const originalSlug = isUpdate ? extractSlugFromUrl(item.originalUrl!) : item.generatedContent.slug;

            setEditedSeo({
                title: item.generatedContent.title,
                metaDescription: item.generatedContent.metaDescription,
                slug: originalSlug,
            });
            setEditedContent(item.generatedContent.content);
            setActiveTab('Live Preview'); // Reset tab on new item
            setWpPublishStatus('idle'); // Reset publish status
            setWpPublishMessage('');
        }
    }, [item]);

    const previewContent = useMemo(() => {
        // The editedContent now contains the base64 images directly, so no replacement is needed for preview.
        return editedContent;
    }, [editedContent]);

    const handleSeoChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        const { name, value } = e.target;
        setEditedSeo(prev => ({ ...prev, [name]: value }));
    };

    const handleSlugChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = e.target.value
            .toLowerCase()
            .replace(/\s+/g, '-') // Replace spaces with -
            .replace(/[^\w-]+/g, ''); // Remove all non-word chars
        setEditedSeo(prev => ({ ...prev, slug: value }));
    };

    const handleCopyHtml = () => {
        if (!item?.generatedContent) return;
        navigator.clipboard.writeText(editedContent)
            .then(() => {
                setCopyStatus('Copied!');
                setTimeout(() => setCopyStatus('Copy HTML'), 2000);
            })
            .catch(err => console.error('Failed to copy HTML: ', err));
    };

    const handleDownloadImage = (base64Data: string, fileName: string) => {
        const link = document.createElement('a');
        link.href = base64Data;
        const safeName = fileName.replace(/[^a-z0-9]/gi, '_').toLowerCase();
        link.download = `${safeName}.jpg`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const handlePublishToWordPress = async () => {
        if (!wpConfig.url || !wpConfig.username || !wpPassword) {
            setWpPublishStatus('error');
            setWpPublishMessage('Please fill in WordPress URL, Username, and Application Password in Step 2.');
            return;
        }

        setWpPublishStatus('publishing');
        
        // Create a temporary item with the latest edits for the publish function
        const itemWithEdits: ContentItem = {
            ...item,
            generatedContent: {
                ...item.generatedContent!,
                ...editedSeo,
                content: editedContent,
            }
        };

        const result = await publishItem(itemWithEdits, wpPassword);

        if (result.success) {
            setWpPublishStatus('success');
            if (item.originalUrl) {
                onPublishSuccess(item.originalUrl);
            }
        } else {
            setWpPublishStatus('error');
        }
        setWpPublishMessage(result.message);
    };

    const TABS = ['Live Preview', 'Editor', 'Assets', 'SEO & Meta', 'Raw JSON'];
    const { title, metaDescription, slug } = editedSeo;
    const { primaryKeyword } = item.generatedContent;

    const titleLength = title.length;
    const titleStatus = titleLength > 60 ? 'bad' : titleLength > 50 ? 'warn' : 'good';
    const metaLength = metaDescription.length;
    const metaStatus = metaLength > 155 ? 'bad' : metaLength > 120 ? 'warn' : 'good';

    const isUpdate = !!item.originalUrl;
    const publishButtonText = isUpdate ? 'Update Live Post' : 'Publish to WordPress';
    const publishingButtonText = isUpdate ? 'Updating...' : 'Publishing...';

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={e => e.stopPropagation()} role="dialog" aria-modal="true" aria-labelledby="review-modal-title">
                <h2 id="review-modal-title" className="sr-only">Review and Edit Content</h2>
                <button className="modal-close-btn" onClick={onClose} aria-label="Close modal">&times;</button>
                <div className="review-tabs" role="tablist">
                    {TABS.map(tab => (
                        <button key={tab} className={`tab-btn ${activeTab === tab ? 'active' : ''}`} onClick={() => setActiveTab(tab)} role="tab" aria-selected={activeTab === tab} aria-controls={`tab-panel-${tab.replace(/\s/g, '-')}`}>
                            {tab}
                        </button>
                    ))}
                </div>

                <div className="tab-content">
                    {activeTab === 'Live Preview' && (
                        <div id="tab-panel-Live-Preview" role="tabpanel" className="live-preview" dangerouslySetInnerHTML={{ __html: previewContent }}></div>
                    )}
                    
                    {activeTab === 'Editor' && (
                        <div id="tab-panel-Editor" role="tabpanel" className="editor-tab-container">
                            <textarea
                                className="html-editor"
                                value={editedContent}
                                onChange={(e) => setEditedContent(e.target.value)}
                                aria-label="HTML Content Editor"
                            />
                        </div>
                    )}

                    {activeTab === 'Assets' && (
                        <div id="tab-panel-Assets" role="tabpanel" className="assets-tab-container">
                            <h3>Generated Images</h3>
                            <p className="help-text" style={{fontSize: '1rem', maxWidth: '800px', margin: '0 0 2rem 0'}}>These images are embedded in your article. They will be automatically uploaded to your WordPress media library when you publish. You can also download them for manual use.</p>
                            <div className="image-assets-grid">
                                {item.generatedContent.imageDetails.map((image, index) => (
                                    image.generatedImageSrc ? (
                                        <div key={index} className="image-asset-card">
                                            <img src={image.generatedImageSrc} alt={image.altText} />
                                            <div className="image-asset-details">
                                                <p><strong>Alt Text:</strong> {image.altText}</p>
                                                <button className="btn btn-small" onClick={() => handleDownloadImage(image.generatedImageSrc!, image.title)}>Download Image</button>
                                            </div>
                                        </div>
                                    ) : null
                                ))}
                            </div>
                        </div>
                    )}

                    {activeTab === 'SEO & Meta' && (
                         <div id="tab-panel-SEO-&-Meta" role="tabpanel" className="seo-meta-container">
                             <div className="seo-meta-tabs" role="tablist" aria-label="SEO & Meta sections">
                                <button className={`seo-meta-tab-btn ${activeSeoTab === 'serp' ? 'active' : ''}`} onClick={() => setActiveSeoTab('serp')} role="tab" aria-selected={activeSeoTab === 'serp'}>
                                    SERP Preview
                                </button>
                                <button className={`seo-meta-tab-btn ${activeSeoTab === 'guardian' ? 'active' : ''}`} onClick={() => setActiveSeoTab('guardian')} role="tab" aria-selected={activeSeoTab === 'guardian'}>
                                    Rank Guardian
                                </button>
                            </div>

                            <div className="seo-meta-grid">
                                <div className="seo-inputs">
                                    <div className="form-group">
                                        <div className="label-wrapper">
                                            <label htmlFor="title">SEO Title</label>
                                            <span className={`char-counter ${titleStatus}`}>{titleLength} / 60</span>
                                        </div>
                                        <input type="text" id="title" name="title" value={title} onChange={handleSeoChange} />
                                        <div className="progress-bar-container">
                                          <div className={`progress-bar-fill ${titleStatus}`} style={{ width: `${Math.min(100, (titleLength / 60) * 100)}%` }}></div>
                                        </div>
                                    </div>
                                    <div className="form-group">
                                         <div className="label-wrapper">
                                            <label htmlFor="metaDescription">Meta Description</label>
                                            <span className={`char-counter ${metaStatus}`}>{metaLength} / 155</span>
                                        </div>
                                        <textarea id="metaDescription" name="metaDescription" className="meta-description-input" value={metaDescription} onChange={handleSeoChange}></textarea>
                                         <div className="progress-bar-container">
                                          <div className={`progress-bar-fill ${metaStatus}`} style={{ width: `${Math.min(100, (metaLength / 155) * 100)}%` }}></div>
                                        </div>
                                    </div>
                                    <div className="form-group">
                                        <label htmlFor="slug">URL Slug</label>
                                        <div className="slug-group">
                                            <span className="slug-base-url">/</span>
                                            <input
                                                type="text"
                                                id="slug"
                                                name="slug"
                                                value={slug}
                                                onChange={handleSlugChange}
                                                disabled={isUpdate}
                                                aria-describedby={isUpdate ? "slug-help" : undefined}
                                            />
                                        </div>
                                        {isUpdate && (
                                            <p id="slug-help" className="help-text" style={{color: 'var(--success-color)'}}>
                                                Original slug is preserved to prevent breaking existing URLs.
                                            </p>
                                        )}
                                    </div>
                                    <div className="form-group">
                                        <label>Primary Keyword</label>
                                        <input type="text" value={primaryKeyword} disabled />
                                    </div>
                                </div>
                                <div className="serp-preview-container">
                                    <h4>Google Preview</h4>
                                    <div className="serp-preview">
                                        <div className="serp-url">{wpConfig.url.replace(/^(https?:\/\/)?/, '').replace(/\/+$/, '')}/{slug}</div>
                                        <a href="#" className="serp-title" onClick={(e) => e.preventDefault()} tabIndex={-1}>{title}</a>
                                        <div className="serp-description">{metaDescription}</div>
                                    </div>
                                </div>
                                {activeSeoTab === 'guardian' && (
                                    <div className="rank-guardian-container">
                                        <RankGuardian seoData={{ ...editedSeo, primaryKeyword, content: editedContent }} />
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {activeTab === 'Raw JSON' && (
                        <pre id="tab-panel-Raw-JSON" role="tabpanel" className="json-viewer">
                            {JSON.stringify(item.generatedContent, null, 2)}
                        </pre>
                    )}
                </div>

                <div className="modal-footer">
                    <div className="wp-publish-container">
                        {wpPublishMessage && <div className={`publish-status ${wpPublishStatus}`} role="alert">{wpPublishMessage}</div>}
                    </div>

                    <div className="modal-actions">
                        <button className="btn btn-secondary" onClick={() => onSaveChanges(item.id, editedSeo, editedContent)}>Save Changes</button>
                        <button className="btn btn-secondary" onClick={handleCopyHtml}>{copyStatus}</button>
                        <button 
                            className="btn btn-success"
                            onClick={handlePublishToWordPress}
                            disabled={wpPublishStatus === 'publishing'}
                        >
                            {wpPublishStatus === 'publishing' ? publishingButtonText : publishButtonText}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

interface BulkPublishModalProps {
    items: ContentItem[];
    onClose: () => void;
    publishItem: (item: ContentItem, password: string) => Promise<{ success: boolean; message: React.ReactNode; link?: string; }>;
    wpPassword: string;
    onPublishSuccess: (originalUrl: string) => void;
}

const BulkPublishModal = ({ items, onClose, publishItem, wpPassword, onPublishSuccess }: BulkPublishModalProps) => {
    const [publishState, setPublishState] = useState<Record<string, { status: 'queued' | 'publishing' | 'success' | 'error', message: React.ReactNode }>>(() => {
        const initialState: Record<string, any> = {};
        items.forEach(item => {
            initialState[item.id] = { status: 'queued', message: 'In queue' };
        });
        return initialState;
    });
    const [isPublishing, setIsPublishing] = useState(false);
    const [isComplete, setIsComplete] = useState(false);

    const handleStartPublishing = async () => {
        setIsPublishing(true);
        setIsComplete(false);
        for (const item of items) {
            setPublishState(prev => ({ ...prev, [item.id]: { status: 'publishing', message: 'Publishing...' } }));
            const result = await publishItem(item, wpPassword);
            setPublishState(prev => ({ ...prev, [item.id]: { status: result.success ? 'success' : 'error', message: result.message } }));
            if (result.success && item.originalUrl) {
                onPublishSuccess(item.originalUrl);
            }
        }
        setIsPublishing(false);
        setIsComplete(true);
    };

    return (
        <div className="modal-overlay" onClick={isPublishing ? undefined : onClose}>
            <div className="modal-content small-modal" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h2>Bulk Publish to WordPress</h2>
                    {!isPublishing && <button className="modal-close-btn" onClick={onClose} aria-label="Close modal">&times;</button>}
                </div>
                <div className="modal-body">
                    <p>The following {items.length} articles will be published sequentially to your WordPress site. Please do not close this window until the process is complete.</p>
                    <ul className="bulk-publish-list">
                        {items.map(item => (
                            <li key={item.id} className="bulk-publish-item">
                                <span className="bulk-publish-item-title" title={item.title}>{item.title}</span>
                                <div className="bulk-publish-item-status">
                                    {publishState[item.id].status === 'queued' && <span style={{ color: 'var(--text-light-color)' }}>Queued</span>}
                                    {publishState[item.id].status === 'publishing' && <><div className="spinner"></div><span>Publishing...</span></>}
                                    {publishState[item.id].status === 'success' && <span className="success">✓ Success</span>}
                                    {publishState[item.id].status === 'error' && <span className="error">✗ Error</span>}
                                </div>
                            </li>
                        ))}
                    </ul>
                     {Object.values(publishState).some(s => s.status === 'error') &&
                        <div className="result error" style={{marginTop: '1.5rem'}}>
                            Some articles failed to publish. Check your WordPress credentials, ensure the REST API is enabled, and try again.
                        </div>
                    }
                </div>
                <div className="modal-footer">
                    {isComplete ? (
                        <button className="btn" onClick={onClose}>Close</button>
                    ) : (
                        <button className="btn" onClick={handleStartPublishing} disabled={isPublishing}>
                            {isPublishing ? `Publishing... (${Object.values(publishState).filter(s => s.status === 'success' || s.status === 'error').length}/${items.length})` : `Publish ${items.length} Articles`}
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
};


// --- Main App Component ---
const App = (): JSX.Element => {
    const [currentStep, setCurrentStep] = useState(1);
    
    // Step 1: API Keys & Config
    const [apiKeys, setApiKeys] = useState(() => {
        const saved = localStorage.getItem('apiKeys');
        return saved ? JSON.parse(saved) : { geminiApiKey: '', openaiApiKey: '', anthropicApiKey: '', openrouterApiKey: '', serperApiKey: '', groqApiKey: '' };
    });
    const [apiKeyStatus, setApiKeyStatus] = useState({ gemini: 'idle', openai: 'idle', anthropic: 'idle', openrouter: 'idle', serper: 'idle', groq: 'idle' } as Record<string, 'idle' | 'validating' | 'valid' | 'invalid'>);
    const [editingApiKey, setEditingApiKey] = useState<string | null>(null);
    const [apiClients, setApiClients] = useState<{ gemini: GoogleGenAI | null, openai: OpenAI | null, anthropic: Anthropic | null, openrouter: OpenAI | null, groq: OpenAI | null }>({ gemini: null, openai: null, anthropic: null, openrouter: null, groq: null });
    const [selectedModel, setSelectedModel] = useState(() => localStorage.getItem('selectedModel') || 'gemini');
    const [selectedGroqModel, setSelectedGroqModel] = useState(() => localStorage.getItem('selectedGroqModel') || AI_MODELS.GROQ_MODELS[0]);
    const [openrouterModels, setOpenrouterModels] = useState<string[]>(AI_MODELS.OPENROUTER_DEFAULT);
    const [geoTargeting, setGeoTargeting] = useState(() => {
        const saved = localStorage.getItem('geoTargeting');
        return saved ? JSON.parse(saved) : { enabled: false, location: '' };
    });
    const [useGoogleSearch, setUseGoogleSearch] = useState(false);


    // Step 2: Content Strategy
    const [contentMode, setContentMode] = useState('bulk'); // 'bulk', 'single', or 'imageGenerator'
    const [topic, setTopic] = useState('');
    const [primaryKeyword, setPrimaryKeyword] = useState('');
    const [sitemapUrl, setSitemapUrl] = useState('');
    const [isCrawling, setIsCrawling] = useState(false);
    const [crawlMessage, setCrawlMessage] = useState('');
    const [crawlProgress, setCrawlProgress] = useState({ current: 0, total: 0 });
    const [existingPages, setExistingPages] = useState<any[]>([]);
    const [wpConfig, setWpConfig] = useState(() => {
        const saved = localStorage.getItem('wpConfig');
        return saved ? JSON.parse(saved) : { url: '', username: '' };
    });
    const [wpPassword, setWpPassword] = useState(() => localStorage.getItem('wpPassword') || '');
    const [wpConnectionStatus, setWpConnectionStatus] = useState<'idle' | 'verifying' | 'valid' | 'invalid'>('idle');
    const [wpConnectionMessage, setWpConnectionMessage] = useState<React.ReactNode>('');


    // Image Generator State
    const [imagePrompt, setImagePrompt] = useState('');
    const [numImages, setNumImages] = useState(1);
    const [aspectRatio, setAspectRatio] = useState('1:1');
    const [isGeneratingImages, setIsGeneratingImages] = useState(false);
    const [generatedImages, setGeneratedImages] = useState<{ src: string, prompt: string }[]>([]); // Array of { src: string, prompt: string }
    const [imageGenerationError, setImageGenerationError] = useState('');

    // Step 3: Generation & Review
    const [items, dispatch] = useReducer(itemsReducer, []);
    const [isGenerating, setIsGenerating] = useState(false);
    const [generationProgress, setGenerationProgress] = useState({ current: 0, total: 0 });
    const [selectedItems, setSelectedItems] = useState(new Set<string>());
    const [filter, setFilter] = useState('');
    const [sortConfig, setSortConfig] = useState({ key: 'title', direction: 'asc' });
    const [selectedItemForReview, setSelectedItemForReview] = useState<ContentItem | null>(null);
    const [isBulkPublishModalOpen, setIsBulkPublishModalOpen] = useState(false);
    const stopGenerationRef = useRef(new Set<string>());
    const isMobile = useMemo(() => window.innerWidth <= 767, []);
    
    // Content Hub State
    const [hubSearchFilter, setHubSearchFilter] = useState('');
    const [hubStatusFilter, setHubStatusFilter] = useState('All');
    const [hubSortConfig, setHubSortConfig] = useState<{key: string, direction: 'asc' | 'desc'}>({ key: 'default', direction: 'desc' });
    const [isAnalyzingHealth, setIsAnalyzingHealth] = useState(false);
    const [healthAnalysisProgress, setHealthAnalysisProgress] = useState({ current: 0, total: 0 });
    const [selectedHubPages, setSelectedHubPages] = useState(new Set<string>());
    
    // Web Worker
    const workerRef = useRef<Worker | null>(null);

    // --- Effects ---
    
    // Persist settings to localStorage
    useEffect(() => { localStorage.setItem('apiKeys', JSON.stringify(apiKeys)); }, [apiKeys]);
    useEffect(() => { localStorage.setItem('selectedModel', selectedModel); }, [selectedModel]);
    useEffect(() => { localStorage.setItem('selectedGroqModel', selectedGroqModel); }, [selectedGroqModel]);
    useEffect(() => { localStorage.setItem('wpConfig', JSON.stringify(wpConfig)); }, [wpConfig]);
    useEffect(() => { localStorage.setItem('wpPassword', wpPassword); }, [wpPassword]);
    useEffect(() => { localStorage.setItem('geoTargeting', JSON.stringify(geoTargeting)); }, [geoTargeting]);


    // Initialize Web Worker
    useEffect(() => {
        const workerCode = `
            self.addEventListener('message', async (e) => {
                const { type, payload } = e.data;

                const fetchWithProxies = ${fetchWithProxies.toString()};
                const extractSlugFromUrl = ${extractSlugFromUrl.toString()};

                if (type === 'CRAWL_SITEMAP') {
                    const { sitemapUrl } = payload;
                    const pageDataMap = new Map();
                    const crawledSitemapUrls = new Set();
                    const sitemapsToCrawl = [sitemapUrl];
                    
                    try {
                        self.postMessage({ type: 'CRAWL_UPDATE', payload: { message: 'Discovering all pages from sitemap(s)...' } });
                        while (sitemapsToCrawl.length > 0) {
                            const currentSitemapUrl = sitemapsToCrawl.shift();
                            if (!currentSitemapUrl || crawledSitemapUrls.has(currentSitemapUrl)) continue;

                            crawledSitemapUrls.add(currentSitemapUrl);
                            self.postMessage({ type: 'CRAWL_UPDATE', payload: { message: \`Parsing sitemap: \${currentSitemapUrl.substring(0, 100)}...\` } });

                            const response = await fetchWithProxies(currentSitemapUrl);
                            const text = await response.text();
                            
                            const initialUrlCount = pageDataMap.size;
                            const sitemapRegex = /<sitemap>\\s*<loc>(.*?)<\\/loc>\\s*<\\/sitemap>/g;
                            const urlBlockRegex = /<url>([\\s\\S]*?)<\\/url>/g;
                            let match;
                            let isSitemapIndex = false;

                            while((match = sitemapRegex.exec(text)) !== null) {
                                sitemapsToCrawl.push(match[1]);
                                isSitemapIndex = true;
                            }

                            while((match = urlBlockRegex.exec(text)) !== null) {
                                const block = match[1];
                                const locMatch = /<loc>(.*?)<\\/loc>/.exec(block);
                                if (locMatch) {
                                    const loc = locMatch[1];
                                    if (!pageDataMap.has(loc)) {
                                        const lastmodMatch = /<lastmod>(.*?)<\\/lastmod>/.exec(block);
                                        const lastmod = lastmodMatch ? lastmodMatch[1] : null;
                                        pageDataMap.set(loc, { lastmod });
                                    }
                                }
                            }

                            if (!isSitemapIndex && pageDataMap.size === initialUrlCount) {
                                self.postMessage({ type: 'CRAWL_UPDATE', payload: { message: \`Using fallback parser for: \${currentSitemapUrl.substring(0, 100)}...\` } });
                                const genericLocRegex = /<loc>(.*?)<\\/loc>/g;
                                while((match = genericLocRegex.exec(text)) !== null) {
                                    const loc = match[1].trim();
                                    if (loc.startsWith('http') && !pageDataMap.has(loc)) {
                                        pageDataMap.set(loc, { lastmod: null });
                                    }
                                }
                            }
                        }

                        const discoveredPages = Array.from(pageDataMap.entries()).map(([url, data]) => {
                            const currentDate = new Date();
                            let daysOld = null;
                            if (data.lastmod) {
                                const lastModDate = new Date(data.lastmod);
                                if (!isNaN(lastModDate.getTime())) {
                                    daysOld = Math.round((currentDate.getTime() - lastModDate.getTime()) / (1000 * 3600 * 24));
                                }
                            }
                            return {
                                id: url,
                                title: url, // Use URL as initial title
                                slug: extractSlugFromUrl(url),
                                lastMod: data.lastmod,
                                wordCount: null,
                                crawledContent: null,
                                healthScore: null,
                                updatePriority: null,
                                justification: null,
                                daysOld: daysOld,
                                isStale: false, // Will be calculated after content analysis
                                publishedState: 'none'
                            };
                        });

                        if (discoveredPages.length === 0) {
                             self.postMessage({ type: 'CRAWL_COMPLETE', payload: { pages: [], message: 'Crawl complete, but no page URLs were found.' } });
                             return;
                        }

                        self.postMessage({ type: 'CRAWL_COMPLETE', payload: { pages: discoveredPages, message: \`Discovery successful! Found \${discoveredPages.length} pages. Click 'Analyze Health' to process content.\` } });

                    } catch (error) {
                        self.postMessage({ type: 'CRAWL_ERROR', payload: { message: \`An error occurred during crawl: \${error.message}\` } });
                    }
                }
            });
        `;
        const blob = new Blob([workerCode], { type: 'application/javascript' });
        workerRef.current = new Worker(URL.createObjectURL(blob));

        workerRef.current.onmessage = (e) => {
            const { type, payload } = e.data;
            switch (type) {
                case 'CRAWL_UPDATE':
                    if (payload.message) setCrawlMessage(payload.message);
                    break;
                case 'CRAWL_COMPLETE':
                    setCrawlMessage(payload.message || 'Crawl complete.');
                    setExistingPages(payload.pages || []);
                    setIsCrawling(false);
                    break;
                case 'CRAWL_ERROR':
                    setCrawlMessage(payload.message);
                    setIsCrawling(false);
                    break;
            }
        };

        return () => {
            workerRef.current?.terminate();
        };
    }, []);

    // Clear hub page selection when filters change to avoid confusion
    useEffect(() => {
        setSelectedHubPages(new Set());
    }, [hubSearchFilter, hubStatusFilter]);

     const filteredAndSortedHubPages = useMemo(() => {
        let filtered = [...existingPages];

        // Status filter
        if (hubStatusFilter !== 'All') {
            filtered = filtered.filter(page => page.updatePriority === hubStatusFilter);
        }

        // Search filter
        if (hubSearchFilter) {
            filtered = filtered.filter(page =>
                page.title.toLowerCase().includes(hubSearchFilter.toLowerCase()) ||
                page.id.toLowerCase().includes(hubSearchFilter.toLowerCase())
            );
        }

        // Sorting
        if (hubSortConfig.key) {
            filtered.sort((a, b) => {
                 if (hubSortConfig.key === 'default') {
                    // 1. Stale content first (true is "smaller" so it comes first with asc)
                    if (a.isStale !== b.isStale) {
                        return a.isStale ? -1 : 1;
                    }
                    // 2. Older content first
                    if (a.daysOld !== b.daysOld) {
                        return (b.daysOld ?? 0) - (a.daysOld ?? 0);
                    }
                    // 3. Thinner content first
                    return (a.wordCount ?? 0) - (b.wordCount ?? 0);
                }

                let valA = a[hubSortConfig.key as keyof typeof a];
                let valB = b[hubSortConfig.key as keyof typeof a];

                // Handle boolean sorting for 'isStale'
                if (typeof valA === 'boolean' && typeof valB === 'boolean') {
                    if (valA === valB) return 0;
                    if (hubSortConfig.direction === 'asc') {
                        return valA ? -1 : 1; // true comes first
                    }
                    return valA ? 1 : -1; // false comes first
                }

                // Handle null or undefined values for sorting
                if (valA === null || valA === undefined) valA = hubSortConfig.direction === 'asc' ? Infinity : -Infinity;
                if (valB === null || valB === undefined) valB = hubSortConfig.direction === 'asc' ? Infinity : -Infinity;

                if (valA < valB) {
                    return hubSortConfig.direction === 'asc' ? -1 : 1;
                }
                if (valA > valB) {
                    return hubSortConfig.direction === 'asc' ? 1 : -1;
                }
                return 0;
            });
        }


        return filtered;
    }, [existingPages, hubSearchFilter, hubStatusFilter, hubSortConfig]);

    const validateApiKey = useCallback(debounce(async (provider: string, key: string) => {
        if (!key) {
            setApiKeyStatus(prev => ({ ...prev, [provider]: 'idle' }));
            setApiClients(prev => ({ ...prev, [provider]: null }));
            return;
        }

        setApiKeyStatus(prev => ({ ...prev, [provider]: 'validating' }));

        try {
            let client;
            let isValid = false;
            switch (provider) {
                case 'gemini':
                    client = new GoogleGenAI({ apiKey: key });
                    await callAiWithRetry(() =>
                        (client as GoogleGenAI).models.generateContent({ model: AI_MODELS.GEMINI_FLASH, contents: 'test' })
                    );
                    isValid = true;
                    break;
                case 'openai':
                    client = new OpenAI({ apiKey: key, dangerouslyAllowBrowser: true });
                    await callAiWithRetry(() => client.models.list());
                    isValid = true;
                    break;
                case 'anthropic':
                    client = new Anthropic({ apiKey: key });
                    await callAiWithRetry(() => client.messages.create({
                        model: AI_MODELS.ANTHROPIC_HAIKU,
                        max_tokens: 1,
                        messages: [{ role: "user", content: "test" }],
                    }));
                    isValid = true;
                    break;
                 case 'openrouter':
                    client = new OpenAI({
                        baseURL: "https://openrouter.ai/api/v1",
                        apiKey: key,
                        dangerouslyAllowBrowser: true,
                        defaultHeaders: {
                            'HTTP-Referer': window.location.href,
                            'X-Title': 'WP Content Optimizer Pro',
                        }
                    });
                    await callAiWithRetry(() => client.chat.completions.create({
                        model: 'google/gemini-2.5-flash',
                        messages: [{ role: "user", content: "test" }],
                        max_tokens: 1
                    }));
                    isValid = true;
                    break;
                case 'groq':
                    client = new OpenAI({
                        baseURL: "https://api.groq.com/openai/v1",
                        apiKey: key,
                        dangerouslyAllowBrowser: true,
                    });
                    await callAiWithRetry(() => (client as OpenAI).chat.completions.create({
                        model: AI_MODELS.GROQ_MODELS[1], // Use a small model for testing
                        messages: [{ role: "user", content: "test" }],
                        max_tokens: 1
                    }));
                    isValid = true;
                    break;
                 case 'serper':
                    const serperResponse = await fetchWithProxies("https://google.serper.dev/search", {
                        method: 'POST',
                        headers: {
                            'X-API-KEY': key,
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ q: 'test' })
                    });
                    if (serperResponse.ok) {
                        isValid = true;
                    } else {
                        const errorBody = await serperResponse.json().catch(() => ({ message: `Serper validation failed with status ${serperResponse.status}` }));
                        throw new Error(errorBody.message || `Serper validation failed with status ${serperResponse.status}`);
                    }
                    break;
            }

            if (isValid) {
                setApiKeyStatus(prev => ({ ...prev, [provider]: 'valid' }));
                if (client) {
                     setApiClients(prev => ({ ...prev, [provider]: client as any }));
                }
                setEditingApiKey(null);
            } else {
                 throw new Error("Validation check failed.");
            }
        } catch (error) {
            console.error(`${provider} API key validation failed:`, error);
            setApiKeyStatus(prev => ({ ...prev, [provider]: 'invalid' }));
            setApiClients(prev => ({ ...prev, [provider]: null }));
        }
    }, 500), []);
    
     useEffect(() => {
        Object.entries(apiKeys).forEach(([key, value]) => {
            if (value) {
                validateApiKey(key.replace('ApiKey', ''), value);
            }
        });
    }, []); // Run only on initial mount to validate saved keys

    const handleApiKeyChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        const { name, value } = e.target;
        const provider = name.replace('ApiKey', '');
        setApiKeys(prev => ({ ...prev, [name]: value }));
        validateApiKey(provider, value);
    };
    
    const handleOpenrouterModelsChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setOpenrouterModels(e.target.value.split('\n').map(m => m.trim()).filter(Boolean));
    };


    const handleNextStep = () => setCurrentStep(prev => prev + 1);
    const handlePrevStep = () => setCurrentStep(prev => prev - 1);

    const handleHubSort = (key: string) => {
        let direction: 'asc' | 'desc' = 'asc';
        if (hubSortConfig.key === key && hubSortConfig.direction === 'asc') {
            direction = 'desc';
        }
        setHubSortConfig({ key, direction });
    };

    const stopHealthAnalysisRef = useRef(false);
    const handleStopHealthAnalysis = () => {
        stopHealthAnalysisRef.current = true;
    };

    const handleAnalyzeContentHealth = async () => {
        const pagesToAnalyze = existingPages.filter(p => !p.crawledContent);
        if (pagesToAnalyze.length === 0) {
            alert("No new pages available to analyze. All discovered pages have already been processed.");
            return;
        }

        const client = apiClients[selectedModel as keyof typeof apiClients];
        if (!client) {
            alert("API client not available. Please check your API key in Step 1.");
            return;
        }
        
        stopHealthAnalysisRef.current = false;
        setIsAnalyzingHealth(true);
        setHealthAnalysisProgress({ current: 0, total: pagesToAnalyze.length });

        try {
            await processConcurrently(
                pagesToAnalyze,
                async (page) => {
                    const cacheKey = `health-analysis-${page.id}`;
                    const cached = sessionStorage.getItem(cacheKey);
                    if(cached) {
                        const parsed = JSON.parse(cached);
                        setExistingPages(prev => prev.map(p => p.id === page.id ? { ...p, ...parsed } : p));
                        return;
                    }

                    try {
                        // --- STAGE 1: Fetch Page Content ---
                        let pageHtml = '';
                        try {
                            const pageResponse = await fetchWithProxies(page.id);
                            pageHtml = await pageResponse.text();
                        } catch (fetchError: any) {
                             throw new Error(`Failed to fetch page content: ${fetchError.message}`);
                        }

                        const titleMatch = pageHtml.match(/<title>([\s\S]*?)<\/title>/i);
                        const title = titleMatch ? titleMatch[1] : 'Untitled Page';

                        let bodyText = pageHtml
                            .replace(/<script[\s\S]*?<\/script>/gi, '')
                            .replace(/<style[\s\S]*?<\/style>/gi, '')
                            .replace(/<nav[\s\S]*?<\/nav>/gi, '')
                            .replace(/<footer[\s\S]*?<\/footer>/gi, '')
                            .replace(/<header[\s\S]*?<\/header>/gi, '')
                            .replace(/<aside[\s\S]*?<\/aside>/gi, '')
                            .replace(/<[^>]+>/g, ' ')
                            .replace(/\s+/g, ' ')
                            .trim();
                        
                        const wordCount = bodyText.split(/\s+/).filter(Boolean).length;
                        
                        const currentYear = new Date().getFullYear();
                        const yearInTitleMatch = title.match(/\b(201[5-9]|202[0-3])\b/);
                        const isStale = yearInTitleMatch ? parseInt(yearInTitleMatch[0], 10) < currentYear : false;

                        setExistingPages(prev => prev.map(p => p.id === page.id ? { ...p, title, wordCount, crawledContent: bodyText, isStale } : p));
                        
                        // --- STAGE 2: AI Health Analysis ---
                        if (wordCount < 100) {
                            throw new Error("Content is too thin for analysis.");
                        }

                        const template = PROMPT_TEMPLATES.content_health_analyzer;
                        const contentSnippet = bodyText.substring(0, 12000); // Use a generous snippet
                        const userPrompt = template.userPrompt(contentSnippet);

                        let responseText: string | null = '';
                        switch (selectedModel) {
                            case 'gemini':
                                const geminiClient = apiClients.gemini;
                                if (!geminiClient) throw new Error("Gemini client not initialized");
                                const geminiResponse = await callAiWithRetry(() => geminiClient.models.generateContent({
                                    model: AI_MODELS.GEMINI_FLASH,
                                    contents: userPrompt,
                                    config: { systemInstruction: template.systemInstruction, responseMimeType: "application/json" }
                                }));
                                responseText = geminiResponse.text;
                                break;
                            case 'openai':
                                const openaiClient = apiClients.openai;
                                if (!openaiClient) throw new Error("OpenAI client not initialized");
                                const openaiResponse = await callAiWithRetry(() => openaiClient.chat.completions.create({
                                    model: AI_MODELS.OPENAI_GPT4_TURBO,
                                    messages: [{ role: "system", content: template.systemInstruction }, { role: "user", content: userPrompt }],
                                    response_format: { type: "json_object" },
                                }));
                                responseText = openaiResponse.choices[0].message.content ?? '';
                                break;
                            case 'openrouter':
                                let openrouterResponseText: string | null = '';
                                let lastError: Error | null = null;
                                const openrouterClient = apiClients.openrouter;
                                if (!openrouterClient) throw new Error("OpenRouter client not initialized");
                                for (const modelName of openrouterModels) {
                                    try {
                                        console.log(`[OpenRouter] Attempting health analysis with model: ${modelName}`);
                                        const response = await callAiWithRetry(() => openrouterClient.chat.completions.create({
                                            model: modelName,
                                            messages: [{ role: "system", content: template.systemInstruction }, { role: "user", content: userPrompt }],
                                            response_format: { type: "json_object" },
                                        }));

                                        const content = response.choices[0].message.content ?? '';
                                        if (!content) throw new Error("Empty response from model.");

                                        extractJson(content);

                                        openrouterResponseText = content;
                                        lastError = null;
                                        break; // Success
                                    } catch (error) {
                                        console.error(`OpenRouter model '${modelName}' failed during health analysis. Trying next...`, error);
                                        lastError = error as Error;
                                    }
                                }
                                if (lastError && !openrouterResponseText) throw lastError;
                                responseText = openrouterResponseText;
                                break;
                            case 'groq':
                                const groqClient = apiClients.groq;
                                if (!groqClient) throw new Error("Groq client not initialized");
                                const groqResponse = await callAiWithRetry(() => groqClient.chat.completions.create({
                                    model: selectedGroqModel,
                                    messages: [{ role: "system", content: template.systemInstruction }, { role: "user", content: userPrompt }],
                                    response_format: { type: "json_object" },
                                }));
                                responseText = groqResponse.choices[0].message.content ?? '';
                                break;
                            case 'anthropic':
                                const anthropicClient = apiClients.anthropic;
                                if (!anthropicClient) throw new Error("Anthropic client not initialized");
                                const anthropicResponse = await callAiWithRetry(() => anthropicClient.messages.create({
                                    model: AI_MODELS.ANTHROPIC_HAIKU,
                                    max_tokens: 4096,
                                    system: template.systemInstruction,
                                    messages: [{ role: "user", content: userPrompt }],
                                }));
                                responseText = anthropicResponse.content.map(c => c.text).join("");
                                break;
                        }

                        const parsedJson = JSON.parse(extractJson(responseText!));
                        const { healthScore, updatePriority, justification } = parsedJson;
                        sessionStorage.setItem(cacheKey, JSON.stringify({ title, wordCount, crawledContent: bodyText, isStale, healthScore, updatePriority, justification }));
                        setExistingPages(prev => prev.map(p => p.id === page.id ? { ...p, healthScore, updatePriority, justification } : p));
                    } catch (error: any) {
                        console.error(`Failed to analyze content for ${page.id}:`, error);
                        setExistingPages(prev => prev.map(p => p.id === page.id ? { ...p, healthScore: 0, updatePriority: 'Error', justification: error.message.substring(0, 100) } : p));
                    }
                },
                8, // Increased concurrency for page fetching
                (completed, total) => {
                    setHealthAnalysisProgress({ current: completed, total: total });
                },
                () => stopHealthAnalysisRef.current
            );
        } catch(error) {
            console.error("Content health analysis process was interrupted or failed:", error);
        } finally {
            setIsAnalyzingHealth(false);
        }
    };


    const handlePlanRewrite = (page: any) => {
        const newItem: ContentItem = { id: page.title, title: page.title, type: 'standard', originalUrl: page.id, status: 'idle', statusText: 'Ready to Rewrite', generatedContent: null, crawledContent: page.crawledContent };
        dispatch({ type: 'SET_ITEMS', payload: [newItem] });
        handleNextStep();
    };

    const handleCreatePillar = (page: any) => {
        const newItem: ContentItem = { id: page.title, title: page.title, type: 'pillar', originalUrl: page.id, status: 'idle', statusText: 'Ready to Generate', generatedContent: null, crawledContent: page.crawledContent };
        dispatch({ type: 'SET_ITEMS', payload: [newItem] });
        handleNextStep();
    };

    const handleToggleHubPageSelect = (pageId: string) => {
        setSelectedHubPages(prev => {
            const newSet = new Set(prev);
            if (newSet.has(pageId)) {
                newSet.delete(pageId);
            } else {
                newSet.add(pageId);
            }
            return newSet;
        });
    };

    const handleToggleHubPageSelectAll = () => {
        if (selectedHubPages.size === filteredAndSortedHubPages.length) {
            setSelectedHubPages(new Set());
        } else {
            setSelectedHubPages(new Set(filteredAndSortedHubPages.map(p => p.id)));
        }
    };

    const handleRewriteSelected = () => {
        const selectedPages = existingPages.filter(p => selectedHubPages.has(p.id));
        if (selectedPages.length === 0) return;

        const newItems: ContentItem[] = selectedPages.map(page => ({
            id: page.title,
            title: page.title,
            type: 'standard',
            originalUrl: page.id,
            status: 'idle',
            statusText: 'Ready to Rewrite',
            generatedContent: null,
            crawledContent: page.crawledContent
        }));
        dispatch({ type: 'SET_ITEMS', payload: newItems });
        setSelectedHubPages(new Set());
        handleNextStep();
    };

    const handleCreatePillarSelected = () => {
        const selectedPages = existingPages.filter(p => selectedHubPages.has(p.id));
        if (selectedPages.length === 0) return;

        const newItems: ContentItem[] = selectedPages.map(page => ({
            id: page.title,
            title: page.title,
            type: 'pillar',
            originalUrl: page.id,
            status: 'idle',
            statusText: 'Ready to Generate',
            generatedContent: null,
            crawledContent: page.crawledContent
        }));
        dispatch({ type: 'SET_ITEMS', payload: newItems });
        setSelectedHubPages(new Set());
        handleNextStep();
    };


    const handleCrawlSitemap = async () => {
        if (!sitemapUrl) {
            setCrawlMessage('Please enter a sitemap URL.');
            return;
        }

        setIsCrawling(true);
        setCrawlMessage('');
        setCrawlProgress({ current: 0, total: 0 });
        setExistingPages([]);
        
        workerRef.current?.postMessage({ type: 'CRAWL_SITEMAP', payload: { sitemapUrl } });
    };
    
    const handleGenerateClusterPlan = async () => {
        setIsGenerating(true);
        dispatch({ type: 'SET_ITEMS', payload: [] });
        const client = apiClients[selectedModel as keyof typeof apiClients];
        if (!client) {
             dispatch({ type: 'UPDATE_STATUS', payload: { id: 'cluster-planner', status: 'error', statusText: 'API Client not initialized.' } });
             setIsGenerating(false);
            return;
        }

        try {
            const template = PROMPT_TEMPLATES.cluster_planner;
            let systemInstruction = template.systemInstruction;
            if (geoTargeting.enabled && geoTargeting.location) {
                systemInstruction = systemInstruction.replace('{{GEO_TARGET_INSTRUCTIONS}}', `All titles must be geo-targeted for "${geoTargeting.location}".`);
            } else {
                 systemInstruction = systemInstruction.replace('{{GEO_TARGET_INSTRUCTIONS}}', '');
            }
            
            const userPrompt = template.userPrompt(topic);
            let responseText: string | null = '';
            
             switch (selectedModel) {
                case 'gemini':
                    const geminiClient = apiClients.gemini;
                    if (!geminiClient) throw new Error("Gemini client not initialized");
                    const geminiResponse = await callAiWithRetry(() => geminiClient.models.generateContent({
                        model: AI_MODELS.GEMINI_FLASH,
                        contents: userPrompt,
                        config: { systemInstruction, responseMimeType: "application/json" }
                    }));
                    responseText = geminiResponse.text;
                    break;
                case 'openai':
                     const openaiClient = apiClients.openai;
                    if (!openaiClient) throw new Error("OpenAI client not initialized");
                    const openaiResponse = await callAiWithRetry(() => openaiClient.chat.completions.create({
                        model: AI_MODELS.OPENAI_GPT4_TURBO,
                        messages: [{ role: "system", content: systemInstruction }, { role: "user", content: userPrompt }],
                        response_format: { type: "json_object" },
                    }));
                    responseText = openaiResponse.choices[0].message.content ?? '';
                    break;
                case 'openrouter':
                    let openrouterResponseText: string | null = '';
                    let lastError: Error | null = null;
                    const openrouterClient = apiClients.openrouter;
                    if (!openrouterClient) throw new Error("OpenRouter client not initialized");
                    
                    for (const modelName of openrouterModels) {
                        try {
                            console.log(`[OpenRouter] Attempting cluster plan with model: ${modelName}`);
                            const response = await callAiWithRetry(() => openrouterClient.chat.completions.create({
                                model: modelName,
                                messages: [{ role: "system", content: systemInstruction }, { role: "user", content: userPrompt }],
                                response_format: { type: "json_object" },
                            }));
                            
                            const content = response.choices[0].message.content ?? '';
                            if (!content) throw new Error("Empty response from model.");
                            
                            extractJson(content); 

                            openrouterResponseText = content;
                            lastError = null;
                            break; // Success
                        } catch (error) {
                            console.error(`OpenRouter model '${modelName}' failed for cluster plan. Trying next...`, error);
                            lastError = error as Error;
                        }
                    }
                    if (lastError && !openrouterResponseText) throw lastError;
                    responseText = openrouterResponseText;
                    break;
                 case 'groq':
                    const groqClient = apiClients.groq;
                    if (!groqClient) throw new Error("Groq client not initialized");
                    const groqResponse = await callAiWithRetry(() => groqClient.chat.completions.create({
                        model: selectedGroqModel,
                        messages: [{ role: "system", content: systemInstruction }, { role: "user", content: userPrompt }],
                        response_format: { type: "json_object" },
                    }));
                    responseText = groqResponse.choices[0].message.content ?? '';
                    break;
                 case 'anthropic':
                    const anthropicClient = apiClients.anthropic;
                    if (!anthropicClient) throw new Error("Anthropic client not initialized");
                    const anthropicResponse = await callAiWithRetry(() => anthropicClient.messages.create({
                        model: AI_MODELS.ANTHROPIC_OPUS,
                        max_tokens: 4096,
                        system: systemInstruction,
                        messages: [{ role: "user", content: userPrompt }],
                    }));
                    responseText = anthropicResponse.content.map(c => c.text).join("");
                    break;
            }

            const parsedJson = JSON.parse(extractJson(responseText!));
            const newItems: Partial<ContentItem>[] = [
                { id: parsedJson.pillarTitle, title: parsedJson.pillarTitle, type: 'pillar' },
                ...parsedJson.clusterTitles.map((title: string) => ({ id: title, title, type: 'cluster' }))
            ];
            dispatch({ type: 'SET_ITEMS', payload: newItems });
            handleNextStep();

        } catch (error: any) {
            console.error("Error generating cluster plan:", error);
            const errorItem: ContentItem = {
                id: 'error-item', title: 'Failed to Generate Plan', type: 'standard', status: 'error',
                statusText: `An error occurred: ${error.message}`, generatedContent: null, crawledContent: null
            };
             dispatch({ type: 'SET_ITEMS', payload: [errorItem] });
        } finally {
            setIsGenerating(false);
        }
    };
    
    const handleGenerateSingleFromKeyword = () => {
        if (!primaryKeyword) return;
        const newItem: Partial<ContentItem> = { id: primaryKeyword, title: primaryKeyword, type: 'standard' };
        dispatch({ type: 'SET_ITEMS', payload: [newItem] });
        handleNextStep();
    };

    // --- Image Generation Logic ---
    const handleGenerateImages = async () => {
        const geminiClient = apiClients.gemini;
        if (!geminiClient || apiKeyStatus.gemini !== 'valid') {
            setImageGenerationError('Please enter a valid Gemini API key in Step 1 to generate images.');
            return;
        }
        if (!imagePrompt) {
            setImageGenerationError('Please enter a prompt to generate an image.');
            return;
        }

        setIsGeneratingImages(true);
        setGeneratedImages([]);
        setImageGenerationError('');

        try {
            const geminiResponse = await callAiWithRetry(() => geminiClient.models.generateImages({
                model: AI_MODELS.GEMINI_IMAGEN,
                prompt: imagePrompt,
                config: {
                    numberOfImages: numImages,
                    outputMimeType: 'image/jpeg',
                    aspectRatio: aspectRatio as "1:1" | "16:9" | "9:16" | "4:3" | "3:4",
                },
            }));
            const imagesData = geminiResponse.generatedImages.map(img => ({
                src: `data:image/jpeg;base64,${img.image.imageBytes}`,
                prompt: imagePrompt
            }));
            
            setGeneratedImages(imagesData);

        } catch (error: any) {
            console.error("Image generation failed:", error);
            setImageGenerationError(`An error occurred: ${error.message}`);
        } finally {
            setIsGeneratingImages(false);
        }
    };

    const handleDownloadImage = (base64Data: string, prompt: string) => {
        const link = document.createElement('a');
        link.href = base64Data;
        const safePrompt = prompt.substring(0, 30).replace(/[^a-z0-9]/gi, '_').toLowerCase();
        link.download = `generated-image-${safePrompt}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const handleCopyText = (text: string) => {
        navigator.clipboard.writeText(text).catch(err => {
            console.error('Failed to copy text: ', err);
        });
    };

    // --- Step 3 Logic ---
    const handleToggleSelect = (itemId: string) => {
        setSelectedItems(prev => {
            const newSet = new Set(prev);
            if (newSet.has(itemId)) {
                newSet.delete(itemId);
            } else {
                newSet.add(itemId);
            }
            return newSet;
        });
    };

    const handleToggleSelectAll = () => {
        if (selectedItems.size === filteredAndSortedItems.length) {
            setSelectedItems(new Set());
        } else {
            setSelectedItems(new Set(filteredAndSortedItems.map(item => item.id)));
        }
    };
    
     const filteredAndSortedItems = useMemo(() => {
        let sorted = [...items];
        if (sortConfig.key) {
            sorted.sort((a, b) => {
                const valA = a[sortConfig.key as keyof typeof a];
                const valB = b[sortConfig.key as keyof typeof a];

                if (valA < valB) {
                    return sortConfig.direction === 'asc' ? -1 : 1;
                }
                if (valA > valB) {
                    return sortConfig.direction === 'asc' ? 1 : -1;
                }
                return 0;
            });
        }
        if (filter) {
            return sorted.filter(item => item.title.toLowerCase().includes(filter.toLowerCase()));
        }
        return sorted;
    }, [items, filter, sortConfig]);

    const handleSort = (key: string) => {
        let direction: 'asc' | 'desc' = 'asc';
        if (sortConfig.key === key && sortConfig.direction === 'asc') {
            direction = 'desc';
        }
        setSortConfig({ key, direction });
    };

    const handleGenerateSingle = (item: ContentItem) => {
        stopGenerationRef.current.delete(item.id);
        setIsGenerating(true);
        setGenerationProgress({ current: 0, total: 1 });
        generateContent([item]);
    };

    const handleGenerateSelected = () => {
        stopGenerationRef.current.clear();
        const itemsToGenerate = items.filter(item => selectedItems.has(item.id));
        if (itemsToGenerate.length > 0) {
            setIsGenerating(true);
            setGenerationProgress({ current: 0, total: itemsToGenerate.length });
            generateContent(itemsToGenerate);
        }
    };
    
     const handleStopGeneration = (itemId: string | null = null) => {
        if (itemId) {
            stopGenerationRef.current.add(itemId);
             dispatch({
                type: 'UPDATE_STATUS',
                payload: { id: itemId, status: 'idle', statusText: 'Stopped by user' }
            });
        } else {
            // Stop all
            items.forEach(item => {
                if (item.status === 'generating') {
                    stopGenerationRef.current.add(item.id);
                     dispatch({
                        type: 'UPDATE_STATUS',
                        payload: { id: item.id, status: 'idle', statusText: 'Stopped by user' }
                    });
                }
            });
            setIsGenerating(false);
        }
    };

    const generateImageWithFallback = async (prompt: string): Promise<string | null> => {
        // Priority 1: OpenAI DALL-E 3
        if (apiClients.openai && apiKeyStatus.openai === 'valid') {
            try {
                console.log("Attempting image generation with OpenAI DALL-E 3...");
                const openaiImgResponse = await callAiWithRetry(() => apiClients.openai!.images.generate({ model: AI_MODELS.OPENAI_DALLE3, prompt, n: 1, size: '1792x1024', response_format: 'b64_json' }));
                if (openaiImgResponse.data && openaiImgResponse.data.length > 0) {
                    const base64Image = openaiImgResponse.data[0].b64_json;
                    if (base64Image) {
                        console.log("OpenAI image generation successful.");
                        return `data:image/png;base64,${base64Image}`;
                    }
                } else {
                     console.warn("OpenAI image generation call succeeded but returned no images. This could be due to a safety filter.");
                }
            } catch (error) {
                console.warn("OpenAI image generation failed, falling back to Gemini.", error);
            }
        }

        // Priority 2: Gemini Imagen
        if (apiClients.gemini && apiKeyStatus.gemini === 'valid') {
            try {
                 console.log("Attempting image generation with Google Gemini Imagen...");
                 const geminiImgResponse = await callAiWithRetry(() => apiClients.gemini!.models.generateImages({ model: AI_MODELS.GEMINI_IMAGEN, prompt: prompt, config: { numberOfImages: 1, outputMimeType: 'image/jpeg', aspectRatio: '16:9' } }));
                 if (geminiImgResponse.generatedImages && geminiImgResponse.generatedImages.length > 0) {
                    const base64Image = geminiImgResponse.generatedImages[0].image.imageBytes;
                    if (base64Image) {
                       console.log("Gemini image generation successful.");
                       return `data:image/jpeg;base64,${base64Image}`;
                    }
                 } else {
                    console.warn("Gemini image generation call succeeded but returned no images. This could be due to a safety filter.");
                 }
            } catch (error) {
                 console.error("Gemini image generation also failed.", error);
            }
        }
        
        console.error("All image generation services failed or are unavailable.");
        try {
            const width = 1792;
            const height = 1024;
            const text = `Image Generation Failed\n\nPrompt: "${prompt.substring(0, 40)}..."`;
            const placeholderUrl = `https://via.placeholder.com/${width}x${height}.png/161b22/c9d1d9?text=${encodeURIComponent(text)}`;
            console.warn(`Falling back to placeholder image: ${placeholderUrl}`);
            return placeholderUrl;
        } catch (e) {
            return null;
        }
    };


    const generateContent = async (itemsToGenerate: ContentItem[]) => {
        const client = apiClients[selectedModel as keyof typeof apiClients];
        if (!client) {
            alert('API Client not initialized. Please check your API key.');
            setIsGenerating(false);
            return;
        }

        let generatedCount = 0;

        for (const item of itemsToGenerate) {
            if (stopGenerationRef.current.has(item.id)) continue;

            dispatch({ type: 'UPDATE_STATUS', payload: { id: item.id, status: 'generating', statusText: 'Initializing...' } });

            let rawResponseForDebugging: any = null;

            try {
                let semanticKeywords: string[] | null = null;
                let serpData: any[] | null = null;
                let paaQuestions: string[] | null = null;
                let relatedSearches: string[] | null = null;
                let youtubeVideos: any[] | null = null;
                
                const isRewrite = !!item.crawledContent;
                const currentSlugForLinking = item.originalUrl ? extractSlugFromUrl(item.originalUrl) : null;

                // --- Universal Pre-flight: Fetch SERP Intelligence for ALL content types ---
                if (apiKeys.serperApiKey && apiKeyStatus.serper === 'valid') {
                    dispatch({ type: 'UPDATE_STATUS', payload: { id: item.id, status: 'generating', statusText: 'Stage 1/4: Fetching SERP Data...' } });
                    const cacheKey = `serp-${item.title}`;
                    const cachedSerp = apiCache.get(cacheKey);

                    if (cachedSerp) {
                         paaQuestions = cachedSerp.paaQuestions;
                         relatedSearches = cachedSerp.relatedSearches;
                         serpData = cachedSerp.serpData;
                         youtubeVideos = cachedSerp.youtubeVideos;
                    } else {
                        try {
                            const serperResponse = await fetchWithProxies("https://google.serper.dev/search", {
                                method: 'POST',
                                headers: { 'X-API-KEY': apiKeys.serperApiKey, 'Content-Type': 'application/json' },
                                body: JSON.stringify({ q: item.title })
                            });
                            if (!serperResponse.ok) throw new Error(`Serper API failed with status ${serperResponse.status}`);
                            const serperJson = await serperResponse.json();

                            paaQuestions = serperJson.peopleAlsoAsk?.map((p: any) => p.question) || null;
                            relatedSearches = serperJson.relatedSearches?.map((r: any) => r.query) || null;
                            serpData = serperJson.organic ? serperJson.organic.slice(0, 10) : [];
                            
                            // --- NEW: Advanced Video Search & Selection ---
                            dispatch({ type: 'UPDATE_STATUS', payload: { id: item.id, status: 'generating', statusText: 'Stage 1/4: Finding Relevant Videos...' } });
                            const videoCandidates = new Map<string, any>();
                            const videoQueries = [`"${item.title}" tutorial`, `how to ${item.title}`, item.title];

                            for (const query of videoQueries) {
                                if (videoCandidates.size >= 10) break;
                                try {
                                    const serperVideoResponse = await fetchWithProxies("https://google.serper.dev/videos", {
                                        method: 'POST',
                                        headers: { 'X-API-KEY': apiKeys.serperApiKey, 'Content-Type': 'application/json' },
                                        body: JSON.stringify({ q: query })
                                    });
                                    if (serperVideoResponse.ok) {
                                        const json = await serperVideoResponse.json();
                                        for (const v of (json.videos || [])) {
                                            const videoId = extractYouTubeID(v.link);
                                            if (videoId && !videoCandidates.has(videoId)) {
                                                videoCandidates.set(videoId, { ...v, videoId });
                                            }
                                        }
                                    }
                                } catch (e) { console.warn(`Video search for query "${query}" failed.`, e); }
                            }

                            const scoredVideos = Array.from(videoCandidates.values()).map(video => {
                                let score = 0;
                                const titleLower = video.title.toLowerCase();
                                const itemTitleLower = item.title.toLowerCase();

                                // Relevance scoring
                                if (titleLower === itemTitleLower) {
                                    score += 25; // Exact match is very strong signal
                                } else if (titleLower.includes(itemTitleLower)) {
                                    score += 15; // Full phrase inclusion
                                }

                                itemTitleLower.split(' ').forEach(word => {
                                    if (word.length > 3 && titleLower.includes(word)) {
                                        score += 2; // Increased score for each keyword
                                    }
                                });

                                // Recency scoring
                                if (video.datePublished) {
                                    const monthsAgo = (new Date().getTime() - new Date(video.datePublished).getTime()) / (1000 * 3600 * 24 * 30);
                                    if (monthsAgo < 6) score += 10;
                                    else if (monthsAgo < 12) score += 5;
                                    else if (monthsAgo < 24) score += 2;
                                }

                                // Clickbait penalty
                                if (/(shocking|secret|exposed|🤯|🔥|must watch)/i.test(video.title)) {
                                    score -= 10;
                                }

                                video.score = score;
                                return video;
                            }).sort((a, b) => b.score - a.score);

                            youtubeVideos = scoredVideos.slice(0, 2).map(v => ({
                                title: v.title,
                                url: v.link,
                                videoId: v.videoId,
                                embedUrl: `https://www.youtube.com/embed/${v.videoId}`
                            }));
                            
                            if ((youtubeVideos?.length ?? 0) < 2) {
                                console.warn(`Could only find ${youtubeVideos?.length ?? 0} highly relevant videos for "${item.title}".`);
                            }

                            apiCache.set(cacheKey, { paaQuestions, relatedSearches, serpData, youtubeVideos });

                        } catch (serpError) {
                            console.error("Failed to fetch SERP data:", serpError);
                            dispatch({ type: 'UPDATE_STATUS', payload: { id: item.id, status: 'generating', statusText: 'SERP fetch failed, continuing...' } });
                        }
                    }
                } else {
                    console.warn("Serper API key not available or invalid. Skipping SERP intelligence fetch.");
                }


                // --- Pre-flight: Generate Semantic Keywords ---
                if (stopGenerationRef.current.has(item.id)) break;
                dispatch({ type: 'UPDATE_STATUS', payload: { id: item.id, status: 'generating', statusText: 'Stage 1/4: Analyzing Topic...' } });
                
                const skCacheKey = `sk-${item.title}`;
                const cachedSk = apiCache.get(skCacheKey);
                if(cachedSk) {
                    semanticKeywords = cachedSk;
                } else {
                    const skTemplate = PROMPT_TEMPLATES.semantic_keyword_generator;
                    const skUserPrompt = skTemplate.userPrompt(item.title);
                    let skResponseText: string | null = '';
                    switch (selectedModel) {
                        case 'gemini':
                            skResponseText = (await callAiWithRetry(() => apiClients.gemini!.models.generateContent({ model: AI_MODELS.GEMINI_FLASH, contents: skUserPrompt, config: { systemInstruction: skTemplate.systemInstruction, responseMimeType: "application/json" } }))).text;
                            break;
                        case 'openai':
                            skResponseText = (await callAiWithRetry(() => apiClients.openai!.chat.completions.create({ model: AI_MODELS.OPENAI_GPT4_TURBO, messages: [{ role: "system", content: skTemplate.systemInstruction }, { role: "user", content: skUserPrompt }], response_format: { type: "json_object" } }))).choices[0].message.content ?? '';
                            break;
                        case 'anthropic':
                            const anthropicResponse = await callAiWithRetry(() => apiClients.anthropic!.messages.create({ model: AI_MODELS.ANTHROPIC_HAIKU, max_tokens: 1024, system: skTemplate.systemInstruction, messages: [{ role: "user", content: skUserPrompt }] }));
                            skResponseText = anthropicResponse.content.map(c => c.text).join("");
                            break;
                        case 'openrouter':
                            // Simplified for brevity - will use first model from list
                            skResponseText = (await callAiWithRetry(() => apiClients.openrouter!.chat.completions.create({ model: openrouterModels[0], messages: [{ role: "system", content: skTemplate.systemInstruction }, { role: "user", content: skUserPrompt }], response_format: { type: "json_object" } }))).choices[0].message.content ?? '';
                            break;
                        case 'groq':
                            const groqResponseSk = await callAiWithRetry(() => apiClients.groq!.chat.completions.create({
                                model: selectedGroqModel,
                                messages: [{ role: "system", content: skTemplate.systemInstruction }, { role: "user", content: skUserPrompt }],
                                response_format: { type: "json_object" }
                            }));
                            skResponseText = groqResponseSk.choices[0].message.content ?? '';
                            break;
                    }
                    const parsedSk = JSON.parse(extractJson(skResponseText!));
                    semanticKeywords = parsedSk.semanticKeywords;
                    apiCache.set(skCacheKey, semanticKeywords);
                }

                // --- Single-shot generation using the new Quantum Quality Protocol ---
                const template = PROMPT_TEMPLATES.premium_content_writer;
                const systemInstruction = (template.systemInstruction as Function)();
                
                const userPrompt = (template.userPrompt as any)(
                    item.title,
                    semanticKeywords,
                    serpData,
                    paaQuestions,
                    relatedSearches,
                    existingPages,
                    youtubeVideos,
                    item.crawledContent, // Pass original content for rewrites
                    currentSlugForLinking // Pass current slug to avoid self-linking
                );

                dispatch({ type: 'UPDATE_STATUS', payload: { id: item.id, status: 'generating', statusText: isRewrite ? 'Stage 2/4: Rewriting Article...' : 'Stage 2/4: Generating Article...' }});
                
                let responseText: string | null = '';
                switch (selectedModel) {
                    case 'gemini':
                        const geminiResponse = await callAiWithRetry(() => apiClients.gemini!.models.generateContent({ model: AI_MODELS.GEMINI_FLASH, contents: userPrompt, config: { systemInstruction: systemInstruction } }));
                        rawResponseForDebugging = geminiResponse;
                        console.log(`[DEBUG] Full Gemini Response for "${item.title}":`, geminiResponse);
                        if ((geminiResponse as any).promptFeedback?.blockReason) {
                            throw new Error(`Content generation blocked by Gemini. Reason: ${(geminiResponse as any).promptFeedback.blockReason}. Details: ${JSON.stringify((geminiResponse as any).promptFeedback.safetyRatings)}`);
                        }
                        responseText = geminiResponse.text;
                        break;
                    case 'openai':
                        const openaiResponse = await callAiWithRetry(() => apiClients.openai!.chat.completions.create({ model: AI_MODELS.OPENAI_GPT4_TURBO, messages: [{ role: "system", content: systemInstruction }, { role: "user", content: userPrompt }] }));
                        rawResponseForDebugging = openaiResponse;
                        console.log(`[DEBUG] Full OpenAI Response for "${item.title}":`, openaiResponse);
                        const choice = openaiResponse.choices[0];
                        if (choice.finish_reason === 'content_filter') {
                            throw new Error('Content generation stopped by OpenAI content filter.');
                        }
                        if (!choice.message.content) {
                            throw new Error(`OpenAI returned no content. Finish Reason: ${choice.finish_reason}.`);
                        }
                        responseText = choice.message.content;
                        break;
                    case 'openrouter':
                        let openrouterResponseText: string | null = null;
                        let lastError: Error | null = null;
                        const openrouterClient = apiClients.openrouter;
                        if (!openrouterClient) throw new Error("OpenRouter client not initialized");
                        
                        const prioritizedModels = [
                            ...openrouterModels.filter(m => !m.includes(':free')),
                            ...openrouterModels.filter(m => m.includes(':free'))
                        ];

                        for (const modelName of prioritizedModels) {
                            if (stopGenerationRef.current.has(item.id)) break;
                            try {
                                console.log(`[OpenRouter] Attempting content generation with model: ${modelName}`);
                                dispatch({ type: 'UPDATE_STATUS', payload: { id: item.id, status: 'generating', statusText: `Stage 2/4: Trying ${modelName.split('/')[1]}...` } });

                                const response = await callAiWithRetry(() => openrouterClient.chat.completions.create({
                                    model: modelName,
                                    messages: [{ role: "system", content: systemInstruction }, { role: "user", content: userPrompt }],
                                }));
                                
                                rawResponseForDebugging = response;
                                const choice = response.choices[0];
                                if (choice.finish_reason === 'content_filter') {
                                    throw new Error('Content generation stopped by OpenRouter content filter.');
                                }
                                if (!choice.message.content) {
                                    throw new Error(`OpenRouter returned no content. Finish Reason: ${choice.finish_reason}.`);
                                }
                                
                                extractJson(choice.message.content);

                                openrouterResponseText = choice.message.content;
                                lastError = null; 
                                console.log(`[OpenRouter] Success with model: ${modelName}`);
                                break; 
                            } catch (error: any) {
                                console.error(`[OpenRouter] Model '${modelName}' failed. Trying next...`, error);
                                lastError = error;
                                if (error.status === 429) {
                                     console.log(`[OpenRouter] Rate limited on model '${modelName}', waiting before trying next...`);
                                     await new Promise(resolve => setTimeout(resolve, 5000));
                                }
                            }
                        }

                        if (lastError && !openrouterResponseText) {
                            throw new Error(`All OpenRouter models failed. Last error: ${lastError.message}`);
                        }
                        responseText = openrouterResponseText;
                        break;
                    case 'groq':
                        const groqResponse = await callAiWithRetry(() => apiClients.groq!.chat.completions.create({
                            model: selectedGroqModel,
                            messages: [{ role: "system", content: systemInstruction }, { role: "user", content: userPrompt }],
                        }));
                        rawResponseForDebugging = groqResponse;
                        const choiceGroq = groqResponse.choices[0];
                        if (choiceGroq.finish_reason === 'content_filter') {
                            throw new Error('Content generation stopped by Groq content filter.');
                        }
                        if (!choiceGroq.message.content) {
                            throw new Error(`Groq returned no content. Finish Reason: ${choiceGroq.finish_reason}.`);
                        }
                        responseText = choiceGroq.message.content;
                        break;
                    case 'anthropic':
                        const anthropicResponse = await callAiWithRetry(() => apiClients.anthropic!.messages.create({ model: AI_MODELS.ANTHROPIC_OPUS, max_tokens: 4096, system: systemInstruction, messages: [{ role: "user", content: userPrompt }] }));
                        rawResponseForDebugging = anthropicResponse;
                        console.log(`[DEBUG] Full Anthropic Response for "${item.title}":`, anthropicResponse);
                        if (anthropicResponse.stop_reason === 'max_tokens') {
                             console.warn("Anthropic response may have been truncated due to max_tokens.");
                        }
                        responseText = anthropicResponse.content.map(c => c.text).join("");
                        break;
                }
                
                if (!responseText) {
                    throw new Error("Received an empty response from the AI model after a successful API call.");
                }

                let finalGeneratedContent = JSON.parse(extractJson(responseText));
                
                // --- START: References Section Purification Protocol ---
                if (finalGeneratedContent.content) {
                    const referencesRegex = /(<h2[^>]*>\s*References\s*<\/h2>)([\s\S]*)/i;
                    const match = finalGeneratedContent.content.match(referencesRegex);
                    
                    if (match) {
                        const h2Tag = match[1];
                        // Replace the entire section from the H2 tag onwards with a purified version
                        finalGeneratedContent.content = finalGeneratedContent.content.substring(0, match.index) + h2Tag + '\n[REFERENCES_PLACEHOLDER]';
                        console.log("[References Purifier] Sanitized References section.");
                    } else {
                        // Safety check: if the AI forgot the H2 entirely.
                        console.warn("[References Purifier] AI forgot References section entirely. Appending it.");
                        finalGeneratedContent.content += '<h2>References</h2>\n[REFERENCES_PLACEHOLDER]';
                    }
                }
                // --- END: References Section Purification Protocol ---

                if (stopGenerationRef.current.has(item.id)) break;
                
                // --- Universal Post-Processing ---
                let processedContent = normalizeGeneratedContent(finalGeneratedContent, item.title);
                
                // --- "ZERO-TOLERANCE" INTERNAL LINKING PROTOCOL ---
                const currentSlug = isRewrite ? extractSlugFromUrl(item.originalUrl!) : processedContent.slug;
                processedContent.content = repairIncorrectAnchorText(processedContent.content, existingPages);
                processedContent.content = validateAndRepairInternalLinks(processedContent.content, existingPages);
                
                const linkTargetCount = Math.floor(Math.random() * (12 - 6 + 1)) + 6;
                processedContent.content = enforceInternalLinkQuota(processedContent.content, existingPages, processedContent.primaryKeyword, linkTargetCount, currentSlug);
                
                processedContent.content = processInternalLinks(processedContent.content, existingPages);
                
                // --- ROBUST IMAGE PLACEHOLDER INJECTION (SAFETY NET) ---
                processedContent.imageDetails.forEach((imgDetail, index) => {
                    if (imgDetail.placeholder && !processedContent.content.includes(imgDetail.placeholder)) {
                        console.warn(`AI did not include placeholder "${imgDetail.placeholder}". Injecting programmatically.`);
                        const paragraphs = processedContent.content.split('</p>');
                        if (paragraphs.length > 2) {
                            const injectionIndex = Math.min(index === 0 ? 2 : 5, paragraphs.length - 1);
                            paragraphs.splice(injectionIndex, 0, `<p>${imgDetail.placeholder}</p>`);
                            processedContent.content = paragraphs.join('</p>');
                        } else {
                            processedContent.content += `<p>${imgDetail.placeholder}</p>`;
                        }
                    }
                });
                
                // --- GUARANTEED VIDEO INJECTION (SAFETY NET 3.0) ---
                if (youtubeVideos && youtubeVideos.length > 0) {
                    const hasVideos = processedContent.content.includes('class="video-container"');
                    if (!hasVideos) {
                        console.warn("AI failed to embed videos. Activating programmatic injection safety net.");
                        let contentWithVideos = processedContent.content;
                        
                        youtubeVideos.slice(0, 2).forEach((video, index) => {
                            const videoHtml = `<h3>${video.title}</h3><div class="video-container"><iframe width="100%" height="410" src="${video.embedUrl}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen title="${video.title}"></iframe></div>`;
                            let injected = false;
                            
                            // Strategy 1: Inject at 30% and 70% points using paragraphs.
                            const paragraphs = contentWithVideos.split('</p>');
                            if (paragraphs.length > 5) {
                                const injectionPoint = Math.floor(paragraphs.length * (index === 0 ? 0.3 : 0.7));
                                paragraphs.splice(injectionPoint, 0, videoHtml);
                                contentWithVideos = paragraphs.join('</p>');
                                injected = true;
                                console.log(`[Video Injector] Success: Injected video ${index + 1} using paragraph strategy.`);
                            }
                            
                            // Strategy 2 (Fallback): Inject after the Nth H2 tag.
                            if (!injected) {
                                const h2matches = [...contentWithVideos.matchAll(/<\/h2>/gi)];
                                const targetH2Index = index === 0 ? 1 : 3;
                                if (h2matches.length > targetH2Index) {
                                    const h2match = h2matches[targetH2Index];
                                    const injectionIndex = h2match.index! + h2match[0].length;
                                    contentWithVideos = contentWithVideos.slice(0, injectionIndex) + videoHtml + contentWithVideos.slice(injectionIndex);
                                    injected = true;
                                    console.log(`[Video Injector] Success: Injected video ${index + 1} using H2 strategy.`);
                                }
                            }
                            
                            // Strategy 3 (Final Failsafe): Append before References section.
                            if (!injected) {
                                const referencesIndex = contentWithVideos.search(/<h2[^>]*>\s*References\s*<\/h2>/i);
                                if (referencesIndex !== -1) {
                                    contentWithVideos = contentWithVideos.substring(0, referencesIndex) + videoHtml + contentWithVideos.substring(referencesIndex);
                                } else {
                                    contentWithVideos += videoHtml; // Absolute last resort
                                }
                                console.log(`[Video Injector] Success: Injected video ${index + 1} using final failsafe.`);
                            }
                        });
                        processedContent.content = contentWithVideos;
                    }
                }
                
                // --- GUARANTEED Reference Injection with EXTERNAL LINK PURIFICATION ---
                let referencesHtml = '';
                if (serpData && serpData.length > 0) {
                    let siteHostname = '';
                    try {
                        if (wpConfig.url) {
                            siteHostname = new URL(wpConfig.url).hostname.replace(/^www\./, '');
                        } else if (sitemapUrl) {
                            siteHostname = new URL(sitemapUrl).hostname.replace(/^www\./, '');
                        }
                    } catch (e) { console.error("Could not parse site hostname for link filtering.", e); }

                    const externalLinks = serpData.filter(ref => {
                        if (!ref.link || !siteHostname) return true;
                        try {
                            const refHostname = new URL(ref.link).hostname.replace(/^www\./, '');
                            return refHostname !== siteHostname;
                        } catch (e) { return true; }
                    });
                    
                    console.log(`[External Links] Found ${serpData.length} total SERP results. Filtered down to ${externalLinks.length} external-only links.`);

                    const numLinks = Math.floor(Math.random() * (8 - 5 + 1)) + 5;
                    const linksToAdd = externalLinks.slice(0, numLinks);

                    if (linksToAdd.length > 0) {
                        referencesHtml = '<ul>';
                        linksToAdd.forEach(ref => {
                            if (ref.link && ref.title) {
                                const context = ref.snippet ? ` - <em class="resource-context">${ref.snippet}</em>` : '';
                                referencesHtml += `<li><a href="${ref.link}" target="_blank" rel="noopener noreferrer">${ref.title}</a>${context}</li>`;
                            }
                        });
                        referencesHtml += '</ul>';
                    } else {
                         referencesHtml = '<p>No relevant external resources were found for this topic.</p>';
                    }
                } else {
                    referencesHtml = '<p>No relevant external resources were found for this topic.</p>';
                }
                processedContent.content = processedContent.content.replace(/\[REFERENCES_PLACEHOLDER\]/g, referencesHtml);
                
                // --- Universal Image Generation with Fallback ---
                for (let i = 0; i < processedContent.imageDetails.length; i++) {
                    const imageDetail = processedContent.imageDetails[i];
                    if (stopGenerationRef.current.has(item.id)) break;
                    dispatch({ type: 'UPDATE_STATUS', payload: { id: item.id, status: 'generating', statusText: `Stage 4/4: Generating Image ${i + 1}/${processedContent.imageDetails.length}...` } });
                    
                    const generatedImageSrc = await generateImageWithFallback(imageDetail.prompt);
                    
                    if (generatedImageSrc) {
                        processedContent.imageDetails[i].generatedImageSrc = generatedImageSrc;
                        const imageHtml = `<figure class="wp-block-image size-large"><img src="${generatedImageSrc}" alt="${imageDetail.altText}" title="${imageDetail.title}"/><figcaption>${imageDetail.altText}</figcaption></figure>`;
                        if (processedContent.content.includes(imageDetail.placeholder)) {
                            processedContent.content = processedContent.content.replace(new RegExp(escapeRegExp(imageDetail.placeholder), 'g'), imageHtml);
                        }
                    } else {
                         if (imageDetail.placeholder) {
                            processedContent.content = processedContent.content.replace(imageDetail.placeholder, `<!-- Image generation failed for prompt: "${imageDetail.prompt}" -->`);
                         }
                    }
                }

                 if (stopGenerationRef.current.has(item.id)) break;

                dispatch({ type: 'UPDATE_STATUS', payload: { id: item.id, status: 'generating', statusText: 'Stage 4/4: Finalizing...' }});
                dispatch({ type: 'SET_CONTENT', payload: { id: item.id, content: processedContent } });

            } catch (error: any) {
                console.error(`Error generating content for "${item.title}":`, error);
                console.log(`[DEBUG] Raw AI Response for "${item.title}":`, rawResponseForDebugging);
                dispatch({ type: 'UPDATE_STATUS', payload: { id: item.id, status: 'error', statusText: error.message } });
            } finally {
                generatedCount++;
                setGenerationProgress({ current: generatedCount, total: itemsToGenerate.length });
            }
        }

        setIsGenerating(false);
    };
    
    // --- WordPress Publishing Logic ---

// FIX: Added explicit return type to satisfy the compiler, which might be confused by the exhaustive switch.
    const getWpConnectionErrorHelp = (errorCode: 'CORS' | 'AUTH' | 'UNREACHABLE' | 'GENERAL', details: string): React.ReactNode => {
        switch (errorCode) {
            case 'CORS':
                return (
                    <div className="cors-instructions">
                        <h4><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z" clipRule="evenodd" /></svg>Connection Blocked by Server (CORS)</h4>
                        <p><strong>Your credentials are likely correct</strong>, but your WordPress server is blocking the connection for security reasons. This is a common server configuration issue that you need to resolve on your website.</p>
                        <p><strong>Primary Solution:</strong></p>
                        <ol>
                            <li>Edit the <strong>.htaccess</strong> file in the main directory of your WordPress installation.</li>
                            <li>Add the following line to the <strong>very top</strong> of the file:
                            <pre><code>SetEnvIf Authorization "(.*)" HTTP_AUTHORIZATION=$1</code></pre>
                            </li>
                        </ol>
                        <p><strong>Other things to check:</strong></p>
                        <ul>
                            <li>Temporarily disable security plugins (like Wordfence, iThemes Security) to see if they are interfering.</li>
                            <li>Ensure your WordPress User is an <strong>Administrator</strong> or <strong>Editor</strong>.</li>
                        </ul>
                    </div>
                );
            case 'AUTH':
                return <p><strong>Authentication Failed (401).</strong> Please double-check your WordPress Username and your Application Password. Ensure the password was copied correctly and is not revoked.</p>;
            case 'UNREACHABLE':
                 return <p><strong>Connection Failed.</strong> Could not reach the WordPress REST API. Please verify your <strong>WordPress URL</strong> is correct and that the site is online. Firewalls or security plugins could also be blocking all access.</p>;
            default:
                 return <p><strong>An unexpected error occurred:</strong> {details}</p>;
        }
    };

    const publishItem = async (
        itemToPublish: ContentItem,
        currentWpPassword: string
    ): Promise<{ success: boolean; message: React.ReactNode; link?: string }> => {
        if (!wpConfig.url || !wpConfig.username || !currentWpPassword) {
            return { success: false, message: 'WordPress URL, Username, and Application Password are required.' };
        }
        if (!itemToPublish.generatedContent) {
            return { success: false, message: 'No content available to publish.' };
        }

        const { url, username } = wpConfig;
        const isUpdate = !!itemToPublish.originalUrl;
        let content = itemToPublish.generatedContent.content;
        const authHeader = `Basic ${btoa(`${username}:${currentWpPassword}`)}`;
        const apiUrlBase = url.replace(/\/+$/, '') + '/wp-json/wp/v2';

        // 1. Upload images and replace base64 src with WordPress URLs
        try {
            for (const imageDetail of itemToPublish.generatedContent.imageDetails) {
                if (imageDetail.generatedImageSrc) {
                    // Convert base64 to Blob
                    const imageResponse = await fetch(imageDetail.generatedImageSrc);
                    const blob = await imageResponse.blob();
                    
                    // Sanitize filename
                    const safeFilename = imageDetail.title
                        .replace(/[^a-z0-9]/gi, '-')
                        .toLowerCase()
                        .substring(0, 50) + '.jpg';

                    const formData = new FormData();
                    formData.append('file', blob, safeFilename);
                    formData.append('title', imageDetail.title);
                    formData.append('alt_text', imageDetail.altText);
                    formData.append('caption', imageDetail.altText);

                    // Use fetchWordPressWithRetry which now correctly handles auth
                    const mediaResponse = await fetchWordPressWithRetry(`${apiUrlBase}/media`, {
                        method: 'POST',
                        headers: {
                            'Authorization': authHeader,
                            // NOTE: Do NOT set 'Content-Type' for FormData, browser does it with boundary
                        },
                        body: formData
                    });

                    if (!mediaResponse.ok) {
                        if (mediaResponse.status === 401) throw new Error('Authentication failed (401). Check credentials.');
                        if (mediaResponse.status === 403) throw new Error('Forbidden (403). User may lack permissions or be blocked by security.');
                        const errorJson = await mediaResponse.json().catch(() => ({}));
                        throw new Error(errorJson.message || `Media upload failed with HTTP ${mediaResponse.status}`);
                    }

                    const mediaJson = await mediaResponse.json();
                    const wpImageUrl = mediaJson.source_url;

                    const newImgTag = `<img src="${wpImageUrl}" alt="${imageDetail.altText}" class="wp-image-${mediaJson.id}" title="${imageDetail.title}"/>`;
                    const oldImgTagRegex = new RegExp(`<img[^>]*src="${escapeRegExp(imageDetail.generatedImageSrc)}"[^>]*>`, 'g');
                    
                    if (oldImgTagRegex.test(content)) {
                        content = content.replace(oldImgTagRegex, newImgTag);
                    } else if (content.includes(imageDetail.placeholder)) {
                        const newFigureBlock = `<figure class="wp-block-image size-large">${newImgTag}<figcaption>${imageDetail.altText}</figcaption></figure>`;
                        content = content.replace(new RegExp(escapeRegExp(imageDetail.placeholder), 'g'), newFigureBlock);
                    }
                }
            }
        } catch (error: any) {
            if (error.name === 'TypeError' || error.message.toLowerCase().includes('cors')) {
                return { success: false, message: getWpConnectionErrorHelp('CORS', error.message) };
            }
            if (error.message.includes('401')) {
                 return { success: false, message: getWpConnectionErrorHelp('AUTH', error.message) };
            }
            return { success: false, message: getWpConnectionErrorHelp('GENERAL', `Media upload failed: ${error.message}`) };
        }

        // 2. Find Post ID for updates
        let postId: number | null = null;
        if (isUpdate) {
            try {
                const slug = extractSlugFromUrl(itemToPublish.originalUrl!);
                const postSearchResponse = await fetchWordPressWithRetry(`${apiUrlBase}/posts?slug=${slug}`, {
                    method: 'GET',
                    headers: { 'Authorization': authHeader }
                });
                 if (!postSearchResponse.ok) {
                    const errorJson = await postSearchResponse.json().catch(() => ({}));
                    throw new Error(errorJson.message || `Could not find existing post (HTTP ${postSearchResponse.status})`);
                }
                const posts = await postSearchResponse.json();
                if (posts.length > 0) {
                    postId = posts[0].id;
                } else {
                    console.warn(`Could not find existing post with slug "${slug}". A new post will be created.`);
                }
            } catch (error: any) {
                return { success: false, message: `Failed to find original post: ${error.message}` };
            }
        }

        // 3. Prepare post data
        const postData = {
            title: itemToPublish.generatedContent.title,
            content: content,
            status: 'publish',
            slug: itemToPublish.generatedContent.slug,
            excerpt: itemToPublish.generatedContent.metaDescription
        };

        // 4. Create or Update Post
        const endpoint = postId ? `${apiUrlBase}/posts/${postId}` : `${apiUrlBase}/posts`;
        try {
            const response = await fetchWordPressWithRetry(endpoint, {
                method: 'POST',
                headers: { 'Authorization': authHeader, 'Content-Type': 'application/json' },
                body: JSON.stringify(postData)
            });

            if (!response.ok) {
                const errorJson = await response.json().catch(() => ({}));
                return { success: false, message: `WP Error: ${errorJson.message || response.statusText}` };
            }

            const responseJson = await response.json();
            const actionText = postId ? 'updated' : 'published';
            const message = (
                <span>
                    Successfully {actionText}!
                    <a href={responseJson.link} target="_blank" rel="noopener noreferrer">
                        View Post
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M4.25 5.5a.75.75 0 00-.75.75v8.5c0 .414.336.75.75.75h8.5a.75.75 0 00.75-.75v-4a.75.75 0 011.5 0v4A2.25 2.25 0 0112.75 17H4.25A2.25 2.25 0 012 14.75v-8.5A2.25 2.25 0 014.25 4h5a.75.75 0 010 1.5h-5z" clipRule="evenodd" /><path fillRule="evenodd" d="M6.194 12.753a.75.75 0 001.06.053L16.5 4.44v2.81a.75.75 0 001.5 0v-4.5a.75.75 0 00-.75-.75h-4.5a.75.75 0 000 1.5h2.553l-9.056 8.19a.75.75 0 00.053 1.06z" clipRule="evenodd" /></svg>
                    </a>
                </span>
            );

            return { success: true, message: message, link: responseJson.link };
        } catch (error: any) {
             if (error.name === 'TypeError' || error.message.toLowerCase().includes('cors')) {
                return { success: false, message: getWpConnectionErrorHelp('CORS', error.message) };
            }
             return { success: false, message: getWpConnectionErrorHelp('GENERAL', error.message) };
        }
    };
    
    const verifyWpConnection = async () => {
        if (!wpConfig.url || !wpConfig.username || !wpPassword) {
            setWpConnectionStatus('invalid');
            setWpConnectionMessage('URL, Username, and Application Password are required.');
            return;
        }
        setWpConnectionStatus('verifying');
        setWpConnectionMessage('');

        const apiUrlBase = wpConfig.url.replace(/\/+$/, '') + '/wp-json';
        const authHeader = `Basic ${btoa(`${wpConfig.username}:${wpPassword}`)}`;

        try {
            // --- STAGE 1: Authenticated Request ---
            const usersResponse = await fetchWordPressWithRetry(`${apiUrlBase}/wp/v2/users/me?context=edit`, {
                method: 'GET',
                headers: { 'Authorization': authHeader }
            });

            if (!usersResponse.ok) {
                if (usersResponse.status === 401) {
                    throw { type: 'AUTH_ERROR', message: 'Authentication failed (401). Incorrect Username or Application Password.' };
                }
                const errorJson = await usersResponse.json().catch(() => ({}));
                throw { type: 'GENERAL_API_ERROR', message: errorJson.message || `API returned HTTP ${usersResponse.status}` };
            }

            const data = await usersResponse.json();
            const canPublish = data.capabilities?.publish_posts;
            if (!canPublish) {
                setWpConnectionStatus('invalid');
                setWpConnectionMessage(`Connected as ${data.name}, but user lacks permission to publish posts. Please use an Administrator or Editor account.`);
                return;
            }

            setWpConnectionStatus('valid');
            setWpConnectionMessage(`Success! Connected as ${data.name} with publish permissions.`);

        } catch (error: any) {
            // --- STAGE 2: Diagnose the Failure ---
            if (error.type === 'AUTH_ERROR') {
                setWpConnectionStatus('invalid');
                setWpConnectionMessage(getWpConnectionErrorHelp('AUTH', error.message));
                return;
            }
            if (error.name === 'TypeError' || error.message?.toLowerCase().includes('failed to fetch')) {
                // This is a network-level error, could be CORS or offline.
                // Let's check if the server is reachable at all without authentication.
                try {
                    const publicResponse = await fetchWordPressWithRetry(apiUrlBase, { method: 'GET' });
                    if (publicResponse.ok) {
                        // Server is online, so the authenticated request was specifically blocked. This is a classic CORS issue.
                        setWpConnectionStatus('invalid');
                        setWpConnectionMessage(getWpConnectionErrorHelp('CORS', error.message));
                    } else {
                        // The server is online but the base REST API endpoint is not working.
                        setWpConnectionStatus('invalid');
                        setWpConnectionMessage(getWpConnectionErrorHelp('UNREACHABLE', `The REST API is not working correctly (HTTP ${publicResponse.status}).`));
                    }
                } catch (publicError) {
                    // We can't even reach the server publicly.
                    setWpConnectionStatus('invalid');
                    setWpConnectionMessage(getWpConnectionErrorHelp('UNREACHABLE', 'The WordPress URL is incorrect or the site is offline.'));
                }
                return;
            }

            // General fallback
            setWpConnectionStatus('invalid');
            setWpConnectionMessage(getWpConnectionErrorHelp('GENERAL', error.message));
        }
    };


    // --- Review Modal Logic ---
    const handleOpenReview = (item: ContentItem) => {
        if (item.generatedContent) {
            setSelectedItemForReview(item);
        }
    };
    
    const handleCloseReview = () => setSelectedItemForReview(null);

    const handleSaveChanges = (itemId: string, updatedSeo: { title: string; metaDescription: string; slug: string; }, updatedContent: string) => {
        const itemToUpdate = items.find(i => i.id === itemId);
        if (itemToUpdate && itemToUpdate.generatedContent) {
            const updatedGeneratedContent: GeneratedContent = {
                ...itemToUpdate.generatedContent,
                ...updatedSeo,
                content: updatedContent
            };
            dispatch({ type: 'SET_CONTENT', payload: { id: itemId, content: updatedGeneratedContent } });
        }
        alert('Changes saved locally!');
    };

    const handlePublishSuccess = (originalUrl: string) => {
        setExistingPages(prev => prev.map(p => 
            p.id === originalUrl ? { ...p, publishedState: 'updated' } : p
        ));
    };

    // --- Render Logic ---
    
    const getWordCountClass = (count: number | null) => {
        if (count === null || count < 0) return '';
        if (count < 800) return 'thin-content';
        return '';
    };

    const getAgeClass = (days: number | null) => {
        if (days === null) return '';
        if (days > 730) return 'age-ancient'; // Over 2 years
        if (days > 365) return 'age-old'; // Over 1 year
        return '';
    };

    const calculateGenerationProgress = (statusText: string): string => {
        if (statusText.includes("Stage 1/4")) return "10%";
        if (statusText.includes("Stage 2/4")) return "35%";
        if (statusText.includes("Stage 3/4")) {
            const match = statusText.match(/Writing Section (\d+)\/(\d+)/);
            if (match) {
                const current = parseInt(match[1], 10);
                const total = parseInt(match[2], 10);
                if (total > 0) {
                    return `${50 + (current / total) * 25}%`;
                }
            }
            return "50%";
        }
        if (statusText.includes("Stage 4/4")) {
            const match = statusText.match(/Generating Image (\d+)\/(\d+)/);
            if (match) {
                const current = parseInt(match[1], 10);
                const total = parseInt(match[2], 10);
                 if (total > 0) {
                    return `${75 + (current / total) * 25}%`;
                }
            }
            if (statusText.includes("Finalizing")) return "99%";
            return "75%";
        }
        return "5%";
    };


    const renderStep = () => {
        switch (currentStep) {
            case 1:
                return (
                    <div className="setup-container">
                        <div className="setup-welcome">
                            <h1 className="usp-headline">Welcome to WP Content Optimizer Pro</h1>
                             <p className="usp-subheadline">
                                Your all-in-one solution for planning, generating, and optimizing high-quality SEO content at scale.
                                Let's get your API keys configured to unlock the power of AI.
                            </p>
                            <div className="features-grid">
                                <div className="feature">
                                    <div className="feature-icon"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" /></svg></div>
                                    <div className="feature-content">
                                        <h3>Multi-Provider AI</h3>
                                        <p>Connect to Gemini, OpenAI, Anthropic, or OpenRouter to use the best models for your needs.</p>
                                    </div>
                                </div>
                                 <div className="feature">
                                    <div className="feature-icon"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.286zm0 13.036h.008v.008h-.008v-.008z" /></svg></div>
                                    <div className="feature-content">
                                        <h3>SEO Optimized</h3>
                                        <p>Every article is crafted based on SEO best practices to help you rank higher on Google.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="setup-form">
                             <fieldset className="config-fieldset">
                                <legend>AI Provider</legend>
                                <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>
                                    <option value="gemini">Google Gemini</option>
                                    <option value="openai">OpenAI</option>
                                    <option value="anthropic">Anthropic</option>
                                    <option value="openrouter">OpenRouter</option>
                                    <option value="groq">Groq</option>
                                </select>
                            </fieldset>
                             <fieldset className="config-fieldset full-width">
                                <legend>SERP Data Provider (Required)</legend>
                                <p className="help-text" style={{margin: '0 0 1rem 0'}}>
                                    An API key from <a href="https://serper.dev" target="_blank" rel="noopener noreferrer">Serper.dev</a> is required for real-time Google data, ensuring content accuracy and high-quality external links.
                                </p>
                                <ApiKeyInput provider="serper" value={apiKeys.serperApiKey} onChange={handleApiKeyChange} status={apiKeyStatus.serper} name="serperApiKey" placeholder="Enter your Serper.dev API Key" isEditing={editingApiKey === 'serper'} onEdit={() => setEditingApiKey('serper')} />
                            </fieldset>
                            <div className="config-forms-wrapper">
                                <fieldset className="config-fieldset">
                                    <legend>Google Gemini</legend>
                                    <ApiKeyInput provider="gemini" value={apiKeys.geminiApiKey} onChange={handleApiKeyChange} status={apiKeyStatus.gemini} name="geminiApiKey" placeholder="Enter your Google Gemini API Key" isEditing={editingApiKey === 'gemini'} onEdit={() => setEditingApiKey('gemini')} />
                                     <p className="help-text">
                                        Used for text and image generation. Recommended for high-quality, cost-effective results.
                                    </p>
                                </fieldset>
                                 <fieldset className="config-fieldset">
                                    <legend>OpenAI</legend>
                                    <ApiKeyInput provider="openai" value={apiKeys.openaiApiKey} onChange={handleApiKeyChange} status={apiKeyStatus.openai} name="openaiApiKey" placeholder="Enter your OpenAI API Key" isEditing={editingApiKey === 'openai'} onEdit={() => setEditingApiKey('openai')} />
                                </fieldset>
                                <fieldset className="config-fieldset">
                                    <legend>Anthropic</legend>
                                    <ApiKeyInput provider="anthropic" value={apiKeys.anthropicApiKey} onChange={handleApiKeyChange} status={apiKeyStatus.anthropic} name="anthropicApiKey" placeholder="Enter your Anthropic API Key" isEditing={editingApiKey === 'anthropic'} onEdit={() => setEditingApiKey('anthropic')} />
                                </fieldset>
                                <fieldset className="config-fieldset">
                                     <legend>OpenRouter</legend>
                                     <ApiKeyInput provider="openrouter" value={apiKeys.openrouterApiKey} onChange={handleApiKeyChange} status={apiKeyStatus.openrouter} name="openrouterApiKey" placeholder="Enter your OpenRouter API Key" isEditing={editingApiKey === 'openrouter'} onEdit={() => setEditingApiKey('openrouter')} />
                                     {selectedModel === 'openrouter' && (
                                         <div className="form-group" style={{marginTop: '1rem'}}>
                                            <label htmlFor="openrouterModels">Custom Model Names (one per line, for fallback)</label>
                                             <ApiKeyInput
                                                provider="openrouterModels"
                                                isTextArea={true}
                                                value={openrouterModels.join('\n')}
                                                onChange={handleOpenrouterModelsChange}
                                                status={'idle'}
                                                name="openrouterModels"
                                                placeholder={"google/gemini-2.5-flash\nanthropic/claude-3-haiku\nmicrosoft/wizardlm-2-8x22b\nopenrouter/auto"}
                                                isEditing={true} onEdit={()=>{}}
                                            />
                                            <p className="help-text" style={{marginTop: '0.5rem'}}>
                                                List models by priority. The tool will try them in order if one fails. Prioritize reliable models over free/unstable ones to avoid errors.
                                            </p>
                                         </div>
                                     )}
                                </fieldset>
                                <fieldset className="config-fieldset">
                                    <legend>Groq</legend>
                                    <ApiKeyInput provider="groq" value={apiKeys.groqApiKey} onChange={handleApiKeyChange} status={apiKeyStatus.groq} name="groqApiKey" placeholder="Enter your Groq API Key" isEditing={editingApiKey === 'groq'} onEdit={() => setEditingApiKey('groq')} />
                                     {selectedModel === 'groq' && (
                                        <div className="form-group" style={{marginTop: '1rem'}}>
                                            <label htmlFor="selectedGroqModel">Groq Model</label>
                                            <input
                                                list="groq-models-list"
                                                id="selectedGroqModel"
                                                name="selectedGroqModel"
                                                value={selectedGroqModel}
                                                onChange={e => setSelectedGroqModel(e.target.value)}
                                                placeholder="Select or type a model name..."
                                            />
                                            <datalist id="groq-models-list">
                                                {AI_MODELS.GROQ_MODELS.map(model => (
                                                    <option key={model} value={model} />
                                                ))}
                                            </datalist>
                                            <p className="help-text" style={{marginTop: '0.5rem'}}>
                                                Select a model from the list or type a custom model ID. Llama 3.1 70B offers the highest quality.
                                            </p>
                                        </div>
                                    )}
                                </fieldset>
                                <fieldset className="config-fieldset full-width">
                                    <legend>Advanced Settings</legend>
                                    <div className="form-group" style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: 0 }}>
                                        <input
                                            type="checkbox"
                                            id="geoEnabled"
                                            checked={geoTargeting.enabled}
                                            onChange={e => setGeoTargeting(p => ({ ...p, enabled: e.target.checked }))}
                                            style={{ width: 'auto' }}
                                        />
                                        <label htmlFor="geoEnabled" style={{ marginBottom: 0, flexGrow: 1 }}>
                                            Enable Geo-Targeting
                                        </label>
                                        <input
                                            type="text"
                                            value={geoTargeting.location}
                                            onChange={e => setGeoTargeting(p => ({ ...p, location: e.target.value }))}
                                            placeholder="e.g., 'New York City'"
                                            disabled={!geoTargeting.enabled}
                                            style={{ flexGrow: 2 }}
                                        />
                                    </div>
                                </fieldset>
                            </div>
                             <button className="btn" onClick={handleNextStep} disabled={apiKeyStatus[selectedModel as keyof typeof apiKeyStatus] !== 'valid' || apiKeyStatus.serper !== 'valid'}>
                                {apiKeyStatus[selectedModel as keyof typeof apiKeyStatus] !== 'valid' ? `Enter a Valid ${selectedModel.charAt(0).toUpperCase() + selectedModel.slice(1)} Key` : apiKeyStatus.serper !== 'valid' ? 'Enter a Valid Serper Key' : 'Proceed to Content Strategy'}
                            </button>
                        </div>
                    </div>
                );
            case 2:
                const priorityOptions = ['All', 'Critical', 'High', 'Medium', 'Healthy', 'Error'];
                return (
                    <div className="step-container full-width">
                        <div className="content-mode-toggle">
                             <button
                                className={contentMode === 'bulk' ? 'active' : ''}
                                onClick={() => setContentMode('bulk')}
                            >
                                Bulk Pillar & Cluster
                            </button>
                            <button
                                className={contentMode === 'single' ? 'active' : ''}
                                onClick={() => setContentMode('single')}
                            >
                                Single Article
                            </button>
                             <button
                                className={contentMode === 'imageGenerator' ? 'active' : ''}
                                onClick={() => setContentMode('imageGenerator')}
                            >
                                AI Image Generator
                            </button>
                        </div>

                        {contentMode === 'imageGenerator' && (
                            <div className="image-generator-container">
                                <div className="content-creation-hub">
                                    <h2>AI Image Generator</h2>
                                    <p>Describe the image you want to create. Be as specific as possible for the best results. Powered by Google's Imagen 4 model.</p>
                                    <div className="image-generator-form">
                                        <div className="form-group">
                                            <textarea
                                                id="imagePrompt"
                                                value={imagePrompt}
                                                onChange={(e) => setImagePrompt(e.target.value)}
                                                placeholder="e.g., A photo of an astronaut riding a horse on Mars, cinematic lighting"
                                                rows={4}
                                            />
                                        </div>
                                        <div className="image-controls-grid">
                                            <div className="form-group">
                                                <label htmlFor="numImages">Number of Images</label>
                                                <select id="numImages" value={numImages} onChange={e => setNumImages(Number(e.target.value))}>
                                                    {[1, 2, 3, 4].map(n => <option key={n} value={n}>{n}</option>)}
                                                </select>
                                            </div>
                                            <div className="form-group">
                                                <label htmlFor="aspectRatio">Aspect Ratio</label>
                                                <select id="aspectRatio" value={aspectRatio} onChange={e => setAspectRatio(e.target.value)}>
                                                    <option value="1:1">Square (1:1)</option>
                                                    <option value="16:9">Widescreen (16:9)</option>
                                                    <option value="9:16">Portrait (9:16)</option>
                                                    <option value="4:3">Landscape (4:3)</option>
                                                    <option value="3:4">Tall (3:4)</option>
                                                </select>
                                            </div>
                                        </div>
                                        <button className="btn" onClick={handleGenerateImages} disabled={isGeneratingImages} style={{width: '100%'}}>
                                            {isGeneratingImages ? 'Generating...' : `Generate ${numImages} Image${numImages > 1 ? 's' : ''}`}
                                        </button>
                                        {imageGenerationError && <p className="result error" style={{textAlign: 'left'}}>{imageGenerationError}</p>}
                                    </div>
                                </div>
                                <div className="image-gallery">
                                    {isGeneratingImages && Array.from({ length: numImages }).map((_, i) => (
                                        <div key={i} className="gallery-placeholder">
                                            <div className="key-status-spinner"></div>
                                            <span>Generating...</span>
                                        </div>
                                    ))}
                                    {generatedImages.map((image, index) => (
                                        <div key={index} className="image-card">
                                            <img src={image.src} alt={image.prompt} />
                                            <div className="image-card-overlay">
                                                <p className="image-card-prompt">{image.prompt}</p>
                                                <div className="image-card-actions">
                                                    <button className="btn btn-small btn-secondary" onClick={() => handleDownloadImage(image.src, image.prompt)}>Download</button>
                                                    <button className="btn btn-small btn-secondary" onClick={() => handleCopyText(image.prompt)}>Copy Prompt</button>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {contentMode === 'bulk' && (
                            <>
                                <div className="content-creation-hub">
                                    <h2>Create a Pillar & Cluster Content Plan</h2>
                                    <p>Enter a broad topic, and we'll generate a complete content plan with a main pillar article and several supporting cluster articles to establish topical authority.</p>
                                    <div className="content-form">
                                        <div className="form-group">
                                            <label htmlFor="topic" className="sr-only">Topic</label>
                                            <input type="text" id="topic" value={topic} onChange={(e) => setTopic(e.target.value)} placeholder="e.g., 'Digital Marketing for Small Businesses'" />
                                        </div>
                                        <button className="btn" onClick={handleGenerateClusterPlan} disabled={isGenerating || !topic}>
                                            {isGenerating ? 'Generating Plan...' : 'Generate Content Plan'}
                                        </button>
                                    </div>
                                </div>
                                <div style={{textAlign: 'center', margin: '2rem 0', fontWeight: 500, color: 'var(--text-light-color)'}}>OR</div>
                                <div className="content-creation-hub content-hub-section">
                                    <div className="content-hub-title" style={{justifyContent: 'center', marginBottom: '0.5rem'}}>
                                        <h3>Content Health Hub</h3>
                                        <div className="tooltip-container">
                                            <span className="info-icon">i</span>
                                            <div className="tooltip-content">
                                                <h4>How It Works</h4>
                                                <p>This tool analyzes your existing content to identify pages that need an update.</p>
                                                <ol>
                                                    <li><strong>Crawl Sitemap:</strong> We'll find all your public pages.</li>
                                                    <li><strong>Analyze Health:</strong> We'll use AI to score each page's SEO health.</li>
                                                    <li><strong>Rewrite or Re-Pillar:</strong> Turn low-scoring content into high-ranking assets.</li>
                                                </ol>
                                            </div>
                                        </div>
                                    </div>
                                    <p>Crawl your website to identify underperforming content and turn it into high-ranking assets.</p>
                                    <div className="sitemap-input-group">
                                        <div className="form-group">
                                            <label htmlFor="sitemapUrl">Sitemap URL</label>
                                            <input type="text" id="sitemapUrl" value={sitemapUrl} onChange={(e) => setSitemapUrl(e.target.value)} placeholder="https://yourwebsite.com/sitemap_index.xml" />
                                        </div>
                                         <button className="btn" onClick={handleCrawlSitemap} disabled={isCrawling}>
                                            {isCrawling ? 'Crawling...' : 'Crawl Sitemap'}
                                        </button>
                                        {crawlMessage && <p className={`result ${crawlMessage.toLowerCase().includes('error') ? 'error' : 'success'}`}>{crawlMessage}</p>}
                                    </div>
                                </div>
                            </>
                        )}

                        {contentMode === 'single' && (
                            <div className="content-creation-hub">
                                <h2>Generate a Single Article</h2>
                                <p>Enter a primary keyword to generate a single, highly-optimized blog post from scratch.</p>
                                <div className="content-form">
                                    <div className="form-group">
                                        <label htmlFor="primaryKeyword" className="sr-only">Primary Keyword</label>
                                        <input type="text" id="primaryKeyword" value={primaryKeyword} onChange={(e) => setPrimaryKeyword(e.target.value)} placeholder="e.g., 'Best SEO Tools for 2025'" />
                                    </div>
                                    <button className="btn" onClick={handleGenerateSingleFromKeyword} disabled={!primaryKeyword}>
                                        Plan Article
                                    </button>
                                </div>
                            </div>
                        )}
                        
                        {(contentMode === 'bulk' || contentMode === 'single') && (
                            <div className="wp-config-section" style={{marginTop: '2rem'}}>
                                <h3>WordPress Integration (Optional)</h3>
                                <p className="help-text">Enter your WordPress details to publish content directly from the app. You must install the <a href="https://wordpress.org/plugins/application-passwords/" target="_blank" rel="noopener noreferrer">Application Passwords</a> plugin to generate a password.</p>
                                <div className="wp-config-grid">
                                    <div className="form-group">
                                        <label htmlFor="wpUrl">WordPress URL</label>
                                        <input type="text" id="wpUrl" value={wpConfig.url} onChange={e => setWpConfig(prev => ({...prev, url: e.target.value}))} placeholder="https://yourwebsite.com" />
                                    </div>
                                    <div className="form-group">
                                        <label htmlFor="wpUsername">WordPress Username</label>
                                        <input type="text" id="wpUsername" value={wpConfig.username} onChange={e => setWpConfig(prev => ({...prev, username: e.target.value}))} placeholder="your-wp-username" />
                                    </div>
                                    <div className="form-group">
                                        <label htmlFor="wpPassword">Application Password</label>
                                        <input type="password" id="wpPassword" value={wpPassword} onChange={e => setWpPassword(e.target.value)} placeholder="xxxx xxxx xxxx xxxx xxxx xxxx" />
                                    </div>
                                </div>
                                <div style={{display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap'}}>
                                    <button className="btn btn-secondary" onClick={verifyWpConnection} disabled={wpConnectionStatus === 'verifying'}>
                                        {wpConnectionStatus === 'verifying' ? 'Verifying...' : 'Verify Connection'}
                                    </button>
                                    {wpConnectionMessage &&
                                        <div className={`wp-connection-status ${wpConnectionStatus}`}>
                                            {wpConnectionStatus === 'verifying' && <div className="spinner"></div>}
                                            {wpConnectionStatus === 'valid' && <svg className="success" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" /></svg>}
                                            {wpConnectionStatus === 'invalid' && <svg className="error" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" /></svg>}
                                            <div className={wpConnectionStatus === 'valid' ? 'success' : 'error'}>{wpConnectionMessage}</div>
                                        </div>
                                    }
                                </div>
                            </div>
                        )}

                        {existingPages.length > 0 && contentMode === 'bulk' && (
                             <div className="content-hub-section" style={{ marginTop: '2rem' }}>
                                <div className="table-toolbar">
                                    <div style={{display: 'flex', alignItems: 'center', gap: '1rem'}}>
                                        <button 
                                            className="btn" 
                                            onClick={handleAnalyzeContentHealth} 
                                            disabled={isAnalyzingHealth}
                                            title="Analyze content for pages that haven't been processed yet"
                                        >
                                            {isAnalyzingHealth ? `Analyzing... (${healthAnalysisProgress.current}/${healthAnalysisProgress.total})` : 'Analyze Content Health'}
                                        </button>
                                        {isAnalyzingHealth && <button className="btn btn-danger" onClick={handleStopHealthAnalysis}>Stop</button>}
                                    </div>
                                    <div className="hub-actions-and-filters">
                                        <div className="hub-filters">
                                            <select className="hub-status-filter" value={hubStatusFilter} onChange={(e) => setHubStatusFilter(e.target.value)}>
                                                {priorityOptions.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                                            </select>
                                            <input
                                                type="search"
                                                className="table-search-input"
                                                placeholder="Filter by title or URL..."
                                                value={hubSearchFilter}
                                                onChange={(e) => setHubSearchFilter(e.target.value)}
                                            />
                                        </div>
                                    </div>
                                </div>
                                
                                {selectedHubPages.size > 0 && (
                                    <div className="bulk-action-bar">
                                        <span>{selectedHubPages.size} page(s) selected</span>
                                        <div className="bulk-action-buttons">
                                            <button className="btn btn-small" onClick={handleRewriteSelected}>Plan Rewrite</button>
                                            <button className="btn btn-small btn-secondary" onClick={handleCreatePillarSelected}>Create Pillar</button>
                                        </div>
                                    </div>
                                )}

                                <div className="table-container">
                                    <table className={`content-table ${isMobile ? 'mobile-cards' : ''}`}>
                                        <thead>
                                            <tr>
                                                <th className="checkbox-cell">
                                                    <input type="checkbox"
                                                        checked={selectedHubPages.size > 0 && selectedHubPages.size === filteredAndSortedHubPages.length}
                                                        onChange={handleToggleHubPageSelectAll}
                                                        aria-label="Select all pages"
                                                    />
                                                </th>
                                                <th className="sortable" onClick={() => handleHubSort('title')}>Post Title</th>
                                                <th className="sortable numeric-cell" onClick={() => handleHubSort('healthScore')}>Health</th>
                                                <th className="sortable numeric-cell" onClick={() => handleHubSort('wordCount')}>Words</th>
                                                <th className="sortable numeric-cell" onClick={() => handleHubSort('daysOld')}>Age</th>
                                                <th className="actions-cell">Actions</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {filteredAndSortedHubPages.map(page => (
                                                <tr key={page.id}>
                                                    <td data-label="Select" className="checkbox-cell">
                                                        <input type="checkbox"
                                                            checked={selectedHubPages.has(page.id)}
                                                            onChange={() => handleToggleHubPageSelect(page.id)}
                                                            aria-label={`Select ${page.title}`}
                                                        />
                                                    </td>
                                                    <td data-label="Post Title">
                                                        <div>
                                                            <a href={page.id} target="_blank" rel="noopener noreferrer" className="hub-post-link">
                                                                <div className="post-title">
                                                                    {page.isStale && <span className="stale-badge priority-high">Outdated</span>}
                                                                    {page.publishedState === 'updated' && <span className="published-badge updated">✓ Updated</span>}
                                                                    {page.title}
                                                                </div>
                                                                <div className="post-url">{page.id}</div>
                                                            </a>
                                                        </div>
                                                    </td>
                                                    <td data-label="Health" className="numeric-cell">
                                                        {page.healthScore !== null ? (
                                                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
                                                                <span className={`stale-badge priority-${page.updatePriority?.toLowerCase()}`}>{page.updatePriority || 'N/A'}</span>
                                                                <span style={{ fontWeight: 600, fontSize: '1rem', marginTop: '0.25rem' }}>{page.healthScore}</span>
                                                            </div>
                                                        ) : page.crawledContent ? (
                                                            'N/A'
                                                        ) : (
                                                            <div className="skeleton-loader" style={{width: '60px', height: '36px', margin: '0 0 0 auto'}}></div>
                                                        )}
                                                    </td>
                                                    <td data-label="Words" className={`numeric-cell ${getWordCountClass(page.wordCount)}`}>
                                                        {page.wordCount !== null ? page.wordCount : <div className="skeleton-loader" style={{width: '50px', height: '20px', margin: '0 0 0 auto'}}></div>}
                                                    </td>
                                                    <td data-label="Age (Days)" className={`numeric-cell ${getAgeClass(page.daysOld)}`}>
                                                        {page.daysOld !== null ? page.daysOld : <div className="skeleton-loader" style={{width: '40px', height: '20px', margin: '0 0 0 auto'}}></div>}
                                                    </td>
                                                    <td data-label="Actions" className="actions-cell">
                                                        {(page.crawledContent && page.updatePriority !== 'Healthy') ? (
                                                            <div className="action-button-group">
                                                                <button className="btn btn-small" onClick={() => handlePlanRewrite(page)}>Rewrite</button>
                                                                <button className="btn btn-small btn-secondary" onClick={() => handleCreatePillar(page)}>Pillar</button>
                                                            </div>
                                                        ) : (
                                                            page.crawledContent ? 'Healthy' : 'Analyze first'
                                                        )}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        )}
                    </div>
                );
            case 3:
                const doneItems = items.filter(item => item.status === 'done').length;
                return (
                    <div className="step-container full-width">
                        <div className="table-toolbar">
                            <div className="selection-toolbar-actions">
                                <button
                                    className="btn"
                                    onClick={handleGenerateSelected}
                                    disabled={isGenerating || selectedItems.size === 0}
                                >
                                    {isGenerating ? 'Generating...' : `Generate Selected (${selectedItems.size})`}
                                </button>
                                {isGenerating && <button className="btn btn-danger" onClick={() => handleStopGeneration()}>Stop All</button>}
                                <button
                                    className="btn btn-secondary"
                                    onClick={() => setIsBulkPublishModalOpen(true)}
                                    disabled={isGenerating || !selectedItems.size || selectedItems.size !== items.filter(i => selectedItems.has(i.id) && i.status === 'done').length}
                                    title={isGenerating ? "Cannot publish while generating" : !selectedItems.size ? "Select items to publish" : "All selected items must be 'Done' to bulk publish"}
                                >
                                    Bulk Publish ({items.filter(i => selectedItems.has(i.id) && i.status === 'done').length})
                                </button>
                            </div>
                             <input
                                type="search"
                                className="table-search-input"
                                placeholder="Filter content..."
                                value={filter}
                                onChange={(e) => setFilter(e.target.value)}
                            />
                        </div>

                         {isGenerating && (
                            <div className="generation-progress-bar">
                                <div className="progress-bar-fill" style={{ width: `${generationProgress.total > 0 ? (generationProgress.current / generationProgress.total) * 100 : 0}%` }}></div>
                                <div className="progress-text">
                                    Generating... ({generationProgress.current}/{generationProgress.total})
                                </div>
                                {isGenerating && <button className="btn btn-danger btn-small stop-generation-btn" onClick={() => handleStopGeneration()}>Stop All</button>}
                            </div>
                        )}
                        <div className="table-container">
                            <table className={`content-table ${isMobile ? 'mobile-cards' : ''}`}>
                                <thead>
                                    <tr>
                                        <th className="checkbox-cell"><input type="checkbox" onChange={handleToggleSelectAll} checked={selectedItems.size > 0 && selectedItems.size === filteredAndSortedItems.length && filteredAndSortedItems.length > 0} /></th>
                                        <th className="sortable" onClick={() => handleSort('title')}>Title <span className={`sort-icon ${sortConfig.key === 'title' ? sortConfig.direction : ''}`}></span></th>
                                        <th>Status</th>
                                        <th className="actions-cell">Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {filteredAndSortedItems.map(item => (
                                        <tr key={item.id} className={`${item.type === 'pillar' ? 'pillar-row' : ''} ${selectedItems.has(item.id) ? 'selected' : ''} ${item.status === 'generating' ? 'is-generating' : ''}`}>
                                            <td data-label="Select" className="checkbox-cell">
                                                <input type="checkbox" checked={selectedItems.has(item.id)} onChange={() => handleToggleSelect(item.id)} />
                                            </td>
                                            <td data-label="Title" style={{ fontWeight: item.type === 'pillar' ? 600 : 400 }}>{item.title}</td>
                                            <td data-label="Status">
                                                {item.status === 'generating' ? (
                                                    <div className="generation-in-progress">
                                                        <div className="row-progress-details">
                                                            <span className="row-status-text">{item.statusText}</span>
                                                            <div className="row-progress-bar-container">
                                                                <div className="row-progress-bar-fill" style={{ width: calculateGenerationProgress(item.statusText) }}></div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                ) : (
                                                    <div className={`status-cell status-${item.status}`}>
                                                        {item.status === 'done' && <span className="status-icon">✓</span>}
                                                        {item.status === 'error' && <span className="status-icon">✗</span>}
                                                        <span>{item.statusText}</span>
                                                    </div>
                                                )}
                                            </td>
                                            <td data-label="Actions" className="actions-cell">
                                                {item.status === 'generating' ? (
                                                     <button className="btn btn-small stop-generation-btn-row" onClick={() => handleStopGeneration(item.id)}>Stop</button>
                                                ) : item.status === 'done' ? (
                                                    <button className="btn btn-small" onClick={() => handleOpenReview(item)}>Review</button>
                                                ) : (
                                                    <button className="btn btn-small" onClick={() => handleGenerateSingle(item)} disabled={isGenerating}>Generate</button>
                                                )}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                );
            default:
                return null;
        }
    };

    return (
        <>
            <header>
                <h1>WP Content Optimizer Pro</h1>
            </header>
            <main className="container">
                <ProgressBar currentStep={currentStep} onStepClick={(step) => {
                    if (!isGenerating) {
                         setCurrentStep(step);
                    }
                }} />
                {renderStep()}
            </main>
             <footer className="app-footer">
                <p>WP Content Optimizer Pro v7.8</p>
                <p>Powered by AI. Please review all generated content for accuracy.</p>
            </footer>
            {selectedItemForReview && (
                <ReviewModal
                    item={selectedItemForReview}
                    onClose={handleCloseReview}
                    onSaveChanges={handleSaveChanges}
                    wpConfig={wpConfig}
                    wpPassword={wpPassword}
                    onPublishSuccess={handlePublishSuccess}
                    publishItem={publishItem}
                />
            )}
            {isBulkPublishModalOpen && (
                 <BulkPublishModal
                    items={items.filter(i => selectedItems.has(i.id) && i.status === 'done')}
                    onClose={() => setIsBulkPublishModalOpen(false)}
                    publishItem={publishItem}
                    wpPassword={wpPassword}
                    onPublishSuccess={handlePublishSuccess}
                />
            )}
        </>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);