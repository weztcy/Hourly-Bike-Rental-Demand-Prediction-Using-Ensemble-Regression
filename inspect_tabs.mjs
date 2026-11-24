import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const USER_DATA = path.join(process.cwd(), '.temp_edge_profile_inspect');

const edgeProc = spawn(EDGE_PATH, [
  '--remote-debugging-port=9223',
  '--headless=new',
  '--disable-gpu',
  `--user-data-dir=${USER_DATA}`,
  '--window-size=1600,1100',
  'http://localhost:8501'
], { stdio: 'ignore' });

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function run() {
  await sleep(3000);
  const res = await fetch('http://127.0.0.1:9223/json');
  const list = await res.json();
  const page = list.find(p => p.type === 'page');
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise(r => ws.onopen = r);

  let id = 1;
  const send = (m, p = {}) => new Promise(res => {
    const cur = id++;
    const handler = (e) => {
      const d = JSON.parse(e.data);
      if (d.id === cur) {
        ws.removeEventListener('message', handler);
        res(d.result);
      }
    };
    ws.addEventListener('message', handler);
    ws.send(JSON.stringify({ id: cur, method: m, params: p }));
  });

  await send('Runtime.enable');
  await sleep(4000);

  const evalRes = await send('Runtime.evaluate', {
    expression: `
      (() => {
        const tabs = Array.from(document.querySelectorAll('[role="tab"], button')).map(el => ({
          tag: el.tagName,
          text: el.innerText,
          role: el.getAttribute('role'),
          id: el.id,
          ariaSelected: el.getAttribute('aria-selected')
        }));
        return JSON.stringify(tabs, null, 2);
      })()
    `,
    returnByValue: true
  });

  console.log('Found tabs/buttons:', evalRes.result?.value);

  edgeProc.kill();
  try { fs.rmSync(USER_DATA, { recursive: true, force: true }); } catch (e) {}
}

run();
