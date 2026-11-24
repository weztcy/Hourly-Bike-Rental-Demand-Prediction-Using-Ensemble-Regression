import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';

const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const USER_DATA = path.join(process.cwd(), '.temp_edge_profile');
const OUT_DIR = path.join(process.cwd(), 'screenshots');

if (!fs.existsSync(OUT_DIR)) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
}

// 1. Launch Edge with remote debugging
console.log('Launching Edge on port 9222...');
const edgeProc = spawn(EDGE_PATH, [
  '--remote-debugging-port=9222',
  '--headless=new',
  '--disable-gpu',
  `--user-data-dir=${USER_DATA}`,
  '--window-size=1600,1100',
  'http://localhost:8501'
], { stdio: 'ignore' });

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function getWsUrl() {
  for (let i = 0; i < 20; i++) {
    try {
      const res = await fetch('http://127.0.0.1:9222/json');
      const list = await res.json();
      const page = list.find(p => p.type === 'page' && p.webSocketDebuggerUrl);
      if (page) return page.webSocketDebuggerUrl;
    } catch (e) {
      // wait
    }
    await sleep(500);
  }
  throw new Error('Could not get debug ws url');
}

class CDP {
  constructor(url) {
    this.ws = new WebSocket(url);
    this.id = 1;
    this.callbacks = new Map();
  }

  async init() {
    await new Promise((resolve, reject) => {
      this.ws.onopen = resolve;
      this.ws.onerror = reject;
    });
    this.ws.onmessage = (evt) => {
      const data = JSON.parse(evt.data);
      if (data.id && this.callbacks.has(data.id)) {
        const { resolve, reject } = this.callbacks.get(data.id);
        this.callbacks.delete(data.id);
        if (data.error) reject(new Error(data.error.message));
        else resolve(data.result);
      }
    };
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = this.id++;
      this.callbacks.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async eval(expr) {
    const res = await this.send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
    return res.result?.value;
  }
}

async function run() {
  try {
    const wsUrl = await getWsUrl();
    console.log('Connected to target via CDP:', wsUrl);
    const cdp = new CDP(wsUrl);
    await cdp.init();

    await cdp.send('Page.enable');
    await cdp.send('Runtime.enable');
    await cdp.send('DOM.enable');

    console.log('Waiting for Streamlit page to stabilize...');
    await sleep(4000);

    // Set viewport
    await cdp.send('Emulation.setDeviceMetricsOverride', {
      width: 1600,
      height: 1100,
      deviceScaleFactor: 2,
      mobile: false
    });

    const tabs = [
      { name: '01_Tab1_Validation_Performance.png', label: 'Validation & Test Performance', index: 0 },
      { name: '02_Tab2_Feature_Breakdown.png', label: 'Feature Breakdown', index: 1 },
      { name: '03_Tab3_System_Reproducibility.png', label: 'System & Reproducibility', index: 2 }
    ];

    for (const t of tabs) {
      console.log(`Switching to tab [${t.label}]...`);
      // Click tab
      const clicked = await cdp.eval(`
        (() => {
          const tabButtons = Array.from(document.querySelectorAll('button[data-baseweb="tab"]'));
          if (tabButtons[${t.index}]) {
            tabButtons[${t.index}].click();
            return true;
          }
          return false;
        })()
      `);
      console.log(`Clicked tab ${t.index}:`, clicked);
      await sleep(1500);

      // Take screenshot
      const shot = await cdp.send('Page.captureScreenshot', { format: 'png', fromSurface: true });
      const filePath = path.join(OUT_DIR, t.name);
      fs.writeFileSync(filePath, Buffer.from(shot.data, 'base64'));
      console.log(`Saved screenshot: ${filePath}`);
    }

    console.log('Done capturing all 3 tabs!');
  } catch (err) {
    console.error('Error during capture:', err);
  } finally {
    try {
      edgeProc.kill();
    } catch (e) {}
    try {
      fs.rmSync(USER_DATA, { recursive: true, force: true });
    } catch (e) {}
  }
}

run();
