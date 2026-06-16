import { Component, OnInit, signal, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

const API = 'http://localhost:8000';

interface Options {
  codebases: string[]; strategies: string[]; methods: string[];
  defaults: { codebase: string; strategy: string; method: string };
}
interface Source { rel_path: string; start_line: number; end_line: number; score: number; }
interface QueryResponse {
  answer: string; error?: string | null; cited_files: string[];
  retrieve_ms: number; generate_ms: number;
  config: { codebase: string; strategy: string; method: string; model: string };
  retrieved: Source[];
}

@Component({
  selector: 'app-root',
  imports: [FormsModule],
  template: `
    <header>
      <h1><span class="mono">codebase-qa-rag</span> <span class="badge">100% local</span></h1>
      <p class="sub">Ask natural-language questions about a codebase. Retrieval and a small
        coder model run entirely on your machine via Ollama.</p>
    </header>
    <main>
      <div class="controls">
        <label>Codebase
          <select [(ngModel)]="codebase">
            @for (c of opts()?.codebases; track c) { <option [value]="c">{{ c }}</option> }
          </select>
        </label>
        <label>Chunking
          <select [(ngModel)]="strategy">
            @for (s of opts()?.strategies; track s) { <option [value]="s">{{ s }}</option> }
          </select>
        </label>
        <label>Retrieval
          <select [(ngModel)]="method">
            @for (m of opts()?.methods; track m) { <option [value]="m">{{ m }}</option> }
          </select>
        </label>
      </div>
      <div class="ask-row">
        <textarea [(ngModel)]="question" placeholder="e.g. How does user authentication work?"></textarea>
        <button (click)="ask()" [disabled]="loading()">{{ loading() ? 'Thinking…' : 'Ask' }}</button>
      </div>

      @if (result(); as r) {
        <section class="panel">
          <h2>Answer</h2>
          <div [innerHTML]="renderAnswer(r)"></div>
          <p class="meta">{{ r.config.model }} · {{ r.config.codebase }}/{{ r.config.strategy }}/{{ r.config.method }}
            · retrieve {{ r.retrieve_ms }} ms · generate {{ r.generate_ms }} ms</p>
        </section>
        <section class="panel">
          <h2>Retrieved sources</h2>
          <ul class="sources">
            @for (s of r.retrieved; track s.rel_path + s.start_line) {
              <li><b>{{ s.rel_path }}</b>:{{ s.start_line }}-{{ s.end_line }}
                <span class="score">{{ s.score }}</span></li>
            }
          </ul>
        </section>
      }
    </main>
  `,
  styles: [`
    :host { display:block; --bg:#0f1115; --panel:#181b22; --border:#2a2f3a; --text:#e6e9ef;
      --muted:#97a0b0; --accent:#5b8cff; --accent2:#ffb454; --ok:#5ad19a;
      background:var(--bg); color:var(--text); min-height:100vh;
      font:15px/1.55 -apple-system,"Segoe UI",Roboto,sans-serif; }
    header, main { max-width:980px; margin:0 auto; padding:0 24px; }
    header { padding-top:28px; }
    h1 { font-size:24px; margin:0; } .mono { font-family:Consolas,monospace; color:var(--accent); }
    .badge { background:var(--panel); border:1px solid var(--border); color:var(--ok);
      border-radius:20px; padding:2px 10px; font-size:12px; margin-left:8px; }
    .sub { color:var(--muted); font-size:14px; }
    .controls { display:flex; gap:12px; flex-wrap:wrap; margin:16px 0 14px; }
    label { display:flex; flex-direction:column; gap:4px; font-size:11px; text-transform:uppercase;
      letter-spacing:.08em; color:var(--muted); }
    select, textarea { background:#1f232c; color:var(--text); border:1px solid var(--border);
      border-radius:8px; padding:9px 10px; font-size:14px; }
    .ask-row { display:flex; gap:12px; align-items:flex-end; }
    textarea { flex:1; min-height:64px; resize:vertical; font-family:inherit; }
    button { background:var(--accent); color:#0b0d12; border:none; border-radius:8px;
      padding:12px 22px; font-weight:600; cursor:pointer; }
    button:disabled { opacity:.5; }
    .panel { background:var(--panel); border:1px solid var(--border); border-radius:12px;
      padding:18px 20px; margin-top:18px; }
    .panel h2 { font-size:12px; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); margin:0 0 10px; }
    .meta { color:var(--muted); font-size:12.5px; }
    .sources { padding:0; margin:0; } .sources li { list-style:none; font-family:Consolas,monospace;
      font-size:13px; color:var(--muted); padding:5px 0; border-bottom:1px dashed var(--border); }
    .sources li b { color:var(--text); } .score { float:right; color:var(--accent2); }
    .cite { background:rgba(91,140,255,.16); color:var(--accent); border:1px solid rgba(91,140,255,.35);
      padding:0 6px; border-radius:5px; font-family:Consolas,monospace; font-size:12.5px; }
    code { background:#11141a; padding:1px 5px; border-radius:4px; font-family:Consolas,monospace; }
  `],
})
export class App implements OnInit {
  private http = inject(HttpClient);
  opts = signal<Options | null>(null);
  result = signal<QueryResponse | null>(null);
  loading = signal(false);
  question = '';
  codebase = 'microblog';
  strategy = 'ast';
  method = 'hybrid';

  ngOnInit(): void {
    this.http.get<Options>(`${API}/options`).subscribe(o => {
      this.opts.set(o);
      this.codebase = o.defaults.codebase;
      this.strategy = o.defaults.strategy;
      this.method = o.defaults.method;
    });
  }

  ask(): void {
    if (!this.question.trim()) return;
    this.loading.set(true);
    this.http.post<QueryResponse>(`${API}/query`, {
      question: this.question, codebase: this.codebase,
      strategy: this.strategy, method: this.method,
    }).subscribe({
      next: r => { this.result.set(r); this.loading.set(false); },
      error: () => this.loading.set(false),
    });
  }

  renderAnswer(r: QueryResponse): string {
    if (r.error) return `<em>Error: ${r.error}</em>`;
    let h = (r.answer || '(empty)')
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    h = h.replace(/\[([A-Za-z0-9_./\\-]+\.[A-Za-z0-9]+)\]/g, '<span class="cite">$1</span>');
    h = h.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
    return h.replace(/\n/g, '<br>');
  }
}
