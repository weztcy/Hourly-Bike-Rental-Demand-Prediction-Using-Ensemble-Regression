import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const USER_DATA = path.join(process.cwd(), '.temp_edge_profile_capture');
const OUT_DIR = path.join(process.cwd(), 'screenshots');

if (!fs.existsSync(OUT_DIR)) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
}

console.log('Launching Edge on port 9224...');
const edgeProc = spawn(EDGE_PATH, [
  '--remote-debugging-port=9224',
  '--headless=new',
  '--disable-gpu',
  `--user-data-dir=${USER_DATA}`,
  '--window-size=1600,1200',
  'http://localhost:8501'
], { stdio: 'ignore' });

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function getWsUrl() {
  for (let i = 0; i < 20; i++) {
    try {
      const res = await fetch('http://127.0.0.1:9224/json');
      const list = await res.json();
      const page = list.find(p => p.type === 'page' && p.webSocketDebuggerUrl);
      if (page) return page.webSocketDebuggerUrl;
    } catch (e) {}
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
      height: 1200,
      deviceScaleFactor: 2,
      mobile: false
    });

    const tabs = [
      { filename: '01_Validation_and_Test_Performance.png', label: 'Validation & Test Performance', tabIndex: 0 },
      { filename: '02_Feature_Breakdown.png', label: 'Feature Breakdown', tabIndex: 1 },
      { filename: '03_System_and_Reproducibility.png', label: 'System & Reproducibility', tabIndex: 2 }
    ];

    for (const t of tabs) {
      console.log('Activating tab: ' + t.label + ' (Index ' + t.tabIndex + ')...');
      const clicked = await cdp.eval(`
        (() => {
          const tabElements = Array.from(document.querySelectorAll('[role="tab"]'));
          if (tabElements[${t.tabIndex}]) {
            tabElements[${t.tabIndex}].click();
            return {
              success: true,
              text: tabElements[${t.tabIndex}].innerText,
              ariaSelected: tabElements[${t.tabIndex}].getAttribute('aria-selected')
            };
          }
          return { success: false };
        })()
      `);
      console.log('Click result:', JSON.stringify(clicked));

      await sleep(2500);

      const shot = await cdp.send('Page.captureScreenshot', {
        format: 'png',
        fromSurface: true
      });
      const filePath = path.join(OUT_DIR, t.filename);
      fs.writeFileSync(filePath, Buffer.from(shot.data, 'base64'));
      console.log('Successfully saved: ' + filePath);
    }

    console.log('\nAll 3 tabs successfully captured!');
  } catch (err) {
    console.error('Error during capture:', err);
  } finally {
    try {
      edgeProc.kill();
    } catch (e) {}
    try {
      fs.rmSync(USER_DATA, { recursive: true, force: true });
      fs.rmSync(path.join(process.cwd(), 'run_all_tab_screenshots.mjs'), { force: true });
    } catch (e) {}
  }
}

run();
