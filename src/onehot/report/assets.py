# Stylesheet shared by every report page. Colours are defined as tokens on :root
# and redefined for dark mode (OS preference or the page's theme toggle)
CSS = """
:root{
  color-scheme:light;
  --page:#f9f9f7;--surface:#fcfcfb;--surface-2:#f3f2ee;
  --ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;
  --line:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
  --accent:#2a78d6;--good:#0ca30c;--good-ink:#006300;--warn:#fab219;--bad:#d03b3b;
  --info-bg:#eef4fd;--good-bg:#edf7ed;--warn-bg:#fdf6e6;--bad-bg:#fbeeee;
  --shadow:0 1px 2px rgba(11,11,11,.04),0 1px 8px rgba(11,11,11,.05);
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){
  color-scheme:dark;
  --page:#0d0d0d;--surface:#1a1a19;--surface-2:#232321;
  --ink:#f0efec;--ink-2:#c3c2b7;--muted:#898781;
  --line:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
  --accent:#3987e5;--good-ink:#0ca30c;
  --info-bg:#16212f;--good-bg:#132113;--warn-bg:#2a2213;--bad-bg:#2b1616;
  --shadow:none;
}}
:root[data-theme="dark"]{
  color-scheme:dark;
  --page:#0d0d0d;--surface:#1a1a19;--surface-2:#232321;
  --ink:#f0efec;--ink-2:#c3c2b7;--muted:#898781;
  --line:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
  --accent:#3987e5;--good-ink:#0ca30c;
  --info-bg:#16212f;--good-bg:#132113;--warn-bg:#2a2213;--bad-bg:#2b1616;
  --shadow:none;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--page);color:var(--ink);
  font:15px/1.65 system-ui,-apple-system,"Segoe UI","Hiragino Sans",sans-serif;-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
code{font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--surface-2);padding:1px 5px;border-radius:4px}
.page-header{max-width:1180px;margin:0 auto;padding:40px 24px 8px}
.page-header h1{font-size:32px;line-height:1.2;margin:6px 0 8px;letter-spacing:-.01em}
.subtitle{color:var(--ink-2);font-size:17px;margin:0 0 12px;max-width:760px}
.crumbs{font-size:13px;color:var(--muted);display:flex;gap:8px}
.crumbs a{color:var(--ink-2)}
.tags{display:flex;flex-wrap:wrap;gap:6px}
.tag{font-size:11.5px;color:var(--ink-2);border:1px solid var(--line);border-radius:999px;padding:1px 9px;background:var(--surface)}
.layout{max-width:1180px;margin:0 auto;padding:8px 24px 48px;display:grid;grid-template-columns:200px minmax(0,1fr);gap:40px}
.toc{position:sticky;top:24px;align-self:start;font-size:13px;padding-top:20px}
.toc-title{color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-size:11px;margin-bottom:8px}
.toc ol{list-style:none;margin:0;padding:0;border-left:1px solid var(--line)}
.toc li a{display:block;padding:4px 0 4px 12px;color:var(--ink-2);margin-left:-1px;border-left:2px solid transparent}
.toc li a.active{color:var(--ink);border-left-color:var(--accent);font-weight:600}
main{min-width:0}
.lead{font-size:16px;color:var(--ink-2);max-width:780px}
.section{margin-top:40px;scroll-margin-top:16px}
.section h2{font-size:21px;margin:0 0 6px;letter-spacing:-.005em}
.section-intro{color:var(--ink-2);margin:0 0 14px;max-width:780px}
.section p{max-width:780px}
.block{margin:16px 0 0;padding:0}
.chart,.circuit,.table-wrap,pre.code{background:var(--surface);border-radius:10px;box-shadow:var(--shadow);border:1px solid var(--ring)}
.chart{padding:8px 8px 0}
figcaption{color:var(--muted);font-size:13px;margin:8px 4px 0;max-width:780px}
.circuit{padding:12px;overflow-x:auto;background:#fff}
.circuit svg{display:block;height:auto}
.circuit-grid{display:grid;grid-template-columns:1fr;gap:16px;margin-top:16px}
.circuit-cell{min-width:0}
.circuit-label{font-size:13px;font-weight:600;color:var(--ink-2);margin:0 0 6px 2px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:16px 0 0}
.stat{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px 16px;box-shadow:var(--shadow)}
.stat-label{font-size:12.5px;color:var(--ink-2)}
.stat-value{font-size:26px;font-weight:600;line-height:1.25;margin-top:2px;display:flex;align-items:center;gap:6px}
.stats.hero .stat-value{font-size:30px}
.stat-hint{font-size:12.5px;color:var(--muted);margin-top:4px}
.tone{display:inline-grid;place-items:center;width:20px;height:20px;border-radius:50%;font-size:12px;color:#fff;flex:none}
.tone-good{background:var(--good)}.tone-bad{background:var(--bad)}.tone-warn{background:var(--warn);color:#0b0b0b}
.callout{display:flex;gap:12px;border-radius:10px;padding:12px 16px;margin:16px 0 0;max-width:820px;border:1px solid var(--ring)}
.callout-icon{flex:none;display:grid;place-items:center;width:22px;height:22px;border-radius:50%;font-size:12px;font-weight:700;color:#fff;margin-top:2px}
.callout-info{background:var(--info-bg)}.callout-info .callout-icon{background:var(--accent)}
.callout-good{background:var(--good-bg)}.callout-good .callout-icon{background:var(--good)}
.callout-warn{background:var(--warn-bg)}.callout-warn .callout-icon{background:var(--warn);color:#0b0b0b}
.callout-bad{background:var(--bad-bg)}.callout-bad .callout-icon{background:var(--bad)}
.table-wrap{overflow:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:7px 12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{position:sticky;top:0;background:var(--surface-2);color:var(--ink-2);font-weight:600;font-size:12.5px;cursor:pointer;user-select:none}
th:hover{color:var(--ink)}
.num{text-align:right;font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover td{background:var(--surface-2)}
th .sort::after{content:"";margin-left:4px;color:var(--muted)}
th[aria-sort="ascending"] .sort::after{content:"▲"}th[aria-sort="descending"] .sort::after{content:"▼"}
pre.code{padding:12px 16px;overflow-x:auto;margin:0}
pre.code code{background:none;padding:0}
footer{max-width:1180px;margin:0 auto;padding:24px;color:var(--muted);font-size:12.5px;border-top:1px solid var(--line)}
.index-header .subtitle{max-width:none}
.index-main{max-width:1180px;margin:0 auto;padding:8px 24px 48px}
.group{margin-top:28px}.group h2{font-size:20px;margin:0 0 12px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}
.card{display:flex;flex-direction:column;background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:18px;box-shadow:var(--shadow);color:var(--ink);transition:transform .12s,border-color .12s}
.card:hover{text-decoration:none;transform:translateY(-2px);border-color:var(--accent)}
.card h3{margin:0 0 4px;font-size:17px}.card p{margin:0 0 12px;color:var(--ink-2);font-size:14px;flex:1}
.mini-stats{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px}
.mini-stat{display:flex;flex-direction:column;font-size:12px;color:var(--muted)}
.mini-stat b{font-size:18px;color:var(--ink);display:flex;gap:4px;align-items:center}
.mini-stat .tone{width:16px;height:16px;font-size:10px}
.card-foot{display:flex;justify-content:space-between;align-items:center;gap:8px}
.card time{font-size:11.5px;color:var(--muted);white-space:nowrap}
.empty{color:var(--muted)}
.theme-toggle{position:fixed;top:14px;right:16px;width:34px;height:34px;border-radius:50%;border:1px solid var(--line);background:var(--surface);color:var(--ink-2);font-size:16px;cursor:pointer;z-index:10}
@media (max-width:860px){.layout{grid-template-columns:1fr;gap:0}.toc{display:none}.page-header h1{font-size:26px}}
@media (max-width:520px){.page-header,.layout,.index-main,footer{padding-left:16px;padding-right:16px}}
"""

# Behaviour shared by every report page: KaTeX rendering, theme toggle that also
# restyles Plotly charts, sortable tables and table-of-contents highlighting
JS = r"""
(function(){
  // KaTeX
  document.addEventListener("DOMContentLoaded",function(){
    if(window.renderMathInElement){renderMathInElement(document.body,{delimiters:[
      {left:"$$",right:"$$",display:true},{left:"\\(",right:"\\)",display:false}],throwOnError:false});}
  });
  // Theme: OS preference, overridable by the toggle; restyle Plotly charts to match.
  var root=document.documentElement, mq=window.matchMedia("(prefers-color-scheme: dark)");
  function isDark(){var t=root.dataset.theme;return t?t==="dark":mq.matches;}
  function restyle(){
    if(!window.Plotly)return;
    var d=isDark(), ink=d?"#c3c2b7":"#52514e", grid=d?"#2c2c2a":"#e1e0d9", axis=d?"#383835":"#c3c2b7";
    document.querySelectorAll(".js-plotly-plot").forEach(function(gd){
      var u={"font.color":ink};
      Object.keys(gd.layout||{}).forEach(function(k){
        if(/^[xy]axis\d*$/.test(k)){u[k+".gridcolor"]=grid;u[k+".linecolor"]=axis;u[k+".zerolinecolor"]=axis;u[k+".title.font.color"]=ink;}
      });
      Plotly.relayout(gd,u);
      var idx=[];(gd.data||[]).forEach(function(t,i){if(t.type==="scatter")idx.push(i);});
      if(idx.length)Plotly.restyle(gd,{"marker.line.color":d?"#1a1a19":"#fcfcfb"},idx);
    });
  }
  document.querySelector(".theme-toggle").addEventListener("click",function(){
    root.dataset.theme=isDark()?"light":"dark";
    try{localStorage.setItem("onehot-theme",root.dataset.theme);}catch(e){}
    restyle();
  });
  mq.addEventListener&&mq.addEventListener("change",restyle);
  window.addEventListener("load",function(){if(isDark())restyle();});
  // Sortable tables
  document.querySelectorAll("table.sortable th").forEach(function(th){
    function sort(){
      var table=th.closest("table"), tbody=table.tBodies[0], col=+th.dataset.col;
      var dir=th.getAttribute("aria-sort")==="ascending"?"descending":"ascending";
      table.querySelectorAll("th").forEach(function(o){o.removeAttribute("aria-sort");});
      th.setAttribute("aria-sort",dir);
      var num=th.classList.contains("num"), rows=Array.from(tbody.rows);
      rows.sort(function(a,b){
        var x=a.cells[col].dataset.sort, y=b.cells[col].dataset.sort;
        var c=num?(parseFloat(x)-parseFloat(y)):x.localeCompare(y,undefined,{numeric:true});
        return dir==="ascending"?c:-c;
      });
      rows.forEach(function(r){tbody.appendChild(r);});
    }
    th.addEventListener("click",sort);
    th.addEventListener("keydown",function(e){if(e.key==="Enter"||e.key===" "){e.preventDefault();sort();}});
  });
  // Highlight the current section in the table of contents.
  var links=document.querySelectorAll(".toc a");
  if(links.length&&"IntersectionObserver" in window){
    var map={};links.forEach(function(a){map[a.getAttribute("href").slice(1)]=a;});
    var obs=new IntersectionObserver(function(es){es.forEach(function(e){
      if(e.isIntersecting){links.forEach(function(a){a.classList.remove("active");});
        var a=map[e.target.id];if(a)a.classList.add("active");}});},{rootMargin:"0px 0px -70% 0px"});
    document.querySelectorAll("section.section").forEach(function(s){obs.observe(s);});
  }
})();
"""
