"""HTML report generation for coffee.py html command.

Produces a self-contained interactive HTML page using Apache ECharts.
"""

import json


def generate_report_html(data):
    """Generate the full HTML report page from pre-computed analysis data.

    Args:
        data: dict with keys:
            title, generated_at, n_coffees,
            umap_frames, umap_weights, coffees,
            archetypes, archetype_names, archetype_colors, archetype_counts,
            correlations, dim_names,
            box_data, box_archetype_groups,
            pca_variance, pca_loadings, pca_n_kept,
            processing_methods,
            explore_pairs,
            superlatives, outliers, typicals

    Returns:
        str: complete HTML page content
    """
    data_json = json.dumps(data, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{data["title"]}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.6.0/echarts.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #1a1a2e;
  color: #e0e0e0;
  line-height: 1.6;
}}
.header {{
  text-align: center;
  padding: 2rem 1rem;
  background: linear-gradient(135deg, #16213e, #0f3460);
  border-bottom: 2px solid #e94560;
}}
.header h1 {{ font-size: 2rem; color: #fff; margin-bottom: 0.3rem; }}
.header p {{ color: #aaa; font-size: 0.9rem; }}
.sticky-bar {{
  position: sticky;
  top: 0;
  z-index: 1000;
  background: #0f3460;
  border-bottom: 1px solid #e94560;
  padding: 0.5rem 1rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1rem;
}}
.stock-count {{
  color: #aaa;
  font-size: 0.8rem;
}}
.section {{
  max-width: 1200px;
  margin: 2rem auto;
  padding: 0 1rem;
}}
.section h2 {{
  font-size: 1.3rem;
  color: #e94560;
  margin-bottom: 0.5rem;
  padding-bottom: 0.3rem;
  border-bottom: 1px solid #333;
}}
.section p.desc {{
  color: #999;
  font-size: 0.85rem;
  margin-bottom: 1rem;
}}
.chart {{
  width: 100%;
  height: 500px;
  background: #16213e;
  border-radius: 8px;
  margin-bottom: 1rem;
}}
.chart-tall {{ height: 700px; }}
.chart-short {{ height: 350px; }}
.controls {{
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 0.5rem;
  flex-wrap: wrap;
}}
.controls label {{ color: #ccc; font-size: 0.85rem; }}
.controls input[type="range"] {{ width: 300px; }}
.controls select {{
  background: #16213e;
  color: #e0e0e0;
  border: 1px solid #444;
  padding: 0.3rem 0.5rem;
  border-radius: 4px;
}}
.toggle-group {{
  display: inline-flex;
  border: 1px solid #444;
  border-radius: 4px;
  overflow: hidden;
}}
.toggle-group input[type="radio"] {{ display: none; }}
.toggle-group label {{
  padding: 0.3rem 0.8rem;
  cursor: pointer;
  background: #16213e;
  color: #aaa;
  font-size: 0.8rem;
  border-right: 1px solid #444;
  transition: background 0.15s, color 0.15s;
}}
.toggle-group label:last-of-type {{ border-right: none; }}
.toggle-group input[type="radio"]:checked + label {{
  background: #e94560;
  color: #fff;
}}
.charts-row {{
  display: flex;
  gap: 1rem;
}}
.charts-row .chart {{
  flex: 1;
  min-width: 0;
}}
.slider-value {{
  color: #e94560;
  font-weight: bold;
  min-width: 3rem;
}}
table.data-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
  background: #16213e;
  border-radius: 8px;
  overflow: hidden;
}}
table.data-table th, table.data-table td {{
  padding: 0.5rem 0.8rem;
  text-align: left;
  border-bottom: 1px solid #2a2a4a;
}}
table.data-table th {{
  background: #0f3460;
  color: #e94560;
  font-weight: 600;
}}
table.data-table tr:hover {{ background: #1f2f50; }}
.footer {{
  text-align: center;
  padding: 2rem;
  color: #666;
  font-size: 0.8rem;
}}
</style>
</head>
<body>
<div class="header">
  <h1>SIP &mdash; Sweet Maria&rsquo;s Flavor Space</h1>
  <p>{data["n_coffees"]} coffees analyzed &middot; Generated {data["generated_at"]}</p>
</div>
<div class="sticky-bar">
  <div class="toggle-group">
    <input type="radio" name="global-stock" id="gs-all" value="all" checked>
    <label for="gs-all">All Coffees</label>
    <input type="radio" name="global-stock" id="gs-instock" value="instock">
    <label for="gs-instock">In Stock Only</label>
  </div>
  <span class="stock-count" id="stock-count-label">{data["n_coffees"]} coffees</span>
</div>

<!-- Section 1: UMAP Flavor Map -->
<div class="section">
  <h2>1. Flavor Map (UMAP)</h2>
  <p class="desc">2D projection of the flavor space. Each dot is a coffee, colored by dominant archetype. Adjust the text-weight slider to blend cupping-note semantics into the layout.</p>
  <div class="controls">
    <label>Text weight:</label>
    <input type="range" id="umap-slider" min="0" max="{len(data["umap_weights"]) - 1}" value="0" step="1">
    <span class="slider-value" id="umap-slider-val">{data["umap_weights"][0]:.2f}</span>
  </div>
  <div id="chart-umap" class="chart chart-tall"></div>
</div>

<!-- Section 2: Archetype Profiles -->
<div class="section">
  <h2>2. Archetype Profiles</h2>
  <p class="desc">Radar charts showing each archetype's deviation from the population mean. Cupping dimensions (left) and flavor intensity (right).</p>
  <div class="charts-row">
    <div id="chart-arch-cupping" class="chart" style="height:500px"></div>
    <div id="chart-arch-flavor" class="chart" style="height:500px"></div>
  </div>
</div>

<!-- Section 3: Catalog Composition -->
<div class="section">
  <h2>3. Catalog Composition</h2>
  <p class="desc">Distribution of coffees by different groupings.</p>
  <div class="controls">
    <div class="toggle-group">
      <input type="radio" name="comp-groupby" id="comp-arch" value="archetype" checked>
      <label for="comp-arch">Archetype</label>
      <input type="radio" name="comp-groupby" id="comp-proc" value="processing">
      <label for="comp-proc">Processing</label>
      <input type="radio" name="comp-groupby" id="comp-region" value="region">
      <label for="comp-region">Region</label>
    </div>
  </div>
  <div id="chart-composition" class="chart chart-short"></div>
</div>

<!-- Section 4: Dimension Correlations -->
<div class="section">
  <h2>4. Dimension Correlations</h2>
  <p class="desc">Pearson correlations between all 22 flavor dimensions. Red = trade-off, blue = move together.</p>
  <div id="chart-correlations" class="chart chart-tall"></div>
</div>

<!-- Section 5: Dimension Distributions -->
<div class="section">
  <h2>5. Dimension Distributions</h2>
  <p class="desc">Box plots showing the spread of each dimension. Cupping scores (0&ndash;10) on the left, flavor intensity (0&ndash;5) on the right.</p>
  <div class="controls">
    <div class="toggle-group">
      <input type="radio" name="box-group" id="box-all" value="all" checked>
      <label for="box-all">All</label>
      <input type="radio" name="box-group" id="box-arch" value="archetype">
      <label for="box-arch">By Archetype</label>
    </div>
  </div>
  <div class="charts-row">
    <div id="chart-dist-cupping" class="chart" style="height:450px"></div>
    <div id="chart-dist-flavor" class="chart" style="height:450px"></div>
  </div>
</div>

<!-- Section 6: PCA Factors -->
<div class="section">
  <h2>6. PCA Factor Analysis</h2>
  <p class="desc">Variance explained per principal component and the dimension loadings that define each factor.</p>
  <div id="chart-pca-var" class="chart chart-short"></div>
  <div id="chart-pca-loadings" class="chart chart-tall"></div>
</div>

<!-- Section 7: Processing Method Signatures -->
<div class="section">
  <h2>7. Processing Method Signatures</h2>
  <p class="desc">Average dimension profiles by processing method. Cupping (left) and flavor intensity (right).</p>
  <div class="charts-row">
    <div id="chart-proc-cupping" class="chart" style="height:500px"></div>
    <div id="chart-proc-flavor" class="chart" style="height:500px"></div>
  </div>
</div>

<!-- Section 8: Dimension Contrast Pairs -->
<div class="section">
  <h2>8. Dimension Contrast Pairs</h2>
  <p class="desc">For each dimension, the high/low pair of coffees that are otherwise most similar &mdash; ideal for isolating a single taste characteristic.</p>
  <div id="chart-explore" class="chart chart-tall"></div>
</div>

<!-- Section 9: Superlatives & Outliers -->
<div class="section">
  <h2>9. Superlatives &amp; Outliers</h2>
  <p class="desc">Extremes of the catalog: highest per dimension, most unusual, most typical.</p>
  <table class="data-table" id="table-superlatives">
    <thead><tr><th>Category</th><th>Coffee</th><th>Value</th></tr></thead>
    <tbody id="superlatives-body"></tbody>
  </table>
</div>

<div class="footer">
  SIP &mdash; Sweet Maria&rsquo;s Inventory Picker
</div>

<script>
const D = {data_json};

// --- Helpers ---
function initChart(id) {{
  return echarts.init(document.getElementById(id), 'dark');
}}

// --- Section 1: UMAP Scatter ---
(function() {{
  const chart = initChart('chart-umap');
  const frames = D.umap_frames;
  const weights = D.umap_weights;
  const coffees = D.coffees;
  let curFrame = frames[0];

  function renderFrame(idx) {{
    const frame = frames[idx];
    curFrame = frame;
    const series = [];
    const archNames = frame.archetype_names;
    const colors = frame.archetype_colors;
    const instockOnly = (typeof window._globalStock !== 'undefined') && window._globalStock === 'instock';
    for (let ai = 0; ai < archNames.length; ai++) {{
      const pts = [];
      for (let i = 0; i < coffees.length; i++) {{
        if (frame.dominant[i] === ai) {{
          if (instockOnly && !coffees[i].in_stock) continue;
          pts.push([frame.x[i], frame.y[i], coffees[i].name, coffees[i].in_stock, coffees[i].score, coffees[i].url, coffees[i].flavors, null, i]);
        }}
      }}
      series.push({{
        name: archNames[ai],
        type: 'scatter',
        data: pts,
        symbolSize: function(d) {{ return d[3] ? 8 : 5; }},
        itemStyle: {{ color: colors[ai], opacity: 0.8 }},
      }});
    }}
    // Archetype centers (★) and contrasts (◆) with connecting lines
    if (frame.centers) {{
      const centerPts = [];
      const contrastPts = [];
      const lineData = [];
      for (let ai = 0; ai < archNames.length; ai++) {{
        const ci = frame.centers[ai];
        const cof = coffees[ci.idx];
        centerPts.push([ci.x, ci.y, cof.name, cof.in_stock, cof.score, cof.url, cof.flavors, archNames[ai] + ' archetype center']);
        const cti = frame.contrasts[ai];
        const ctcof = coffees[cti.idx];
        contrastPts.push([cti.x, cti.y, ctcof.name, ctcof.in_stock, ctcof.score, ctcof.url, ctcof.flavors, archNames[ai] + ' contrast pair']);
        lineData.push([{{ coord: [ci.x, ci.y] }}, {{ coord: [cti.x, cti.y] }}]);
      }}
      series.push({{
        name: '\\u2605 Archetype Centers',
        type: 'scatter',
        data: centerPts,
        symbol: 'diamond',
        symbolSize: 18,
        itemStyle: {{ color: '#fff', borderColor: '#fff', borderWidth: 2, opacity: 0.95 }},
        z: 10
      }});
      series.push({{
        name: '\\u25c6 Contrast Pairs',
        type: 'scatter',
        data: contrastPts,
        symbol: 'triangle',
        symbolSize: 14,
        itemStyle: {{ color: '#fff', borderColor: '#fff', borderWidth: 1, opacity: 0.85 }},
        z: 10
      }});
      series.push({{
        name: 'Center\\u2194Contrast',
        type: 'lines',
        coordinateSystem: 'cartesian2d',
        data: lineData,
        lineStyle: {{ color: '#fff', opacity: 0.3, width: 1, type: 'dashed' }},
        z: 5
      }});
    }}
    chart.setOption({{
      tooltip: {{
        trigger: 'item',
        formatter: function(p) {{
          if (!p.data || !Array.isArray(p.data)) return null;
          const d = p.data;
          if (d.length < 6) return null;
          const stock = d[3] ? '\\u2705 In Stock' : '\\u274c Out of Stock';
          const flavors = d[6] ? '<br><b>Scores:</b> ' + d[6] : '';
          const role = d[7] ? '<div style="color:#e94560;font-size:11px;margin-bottom:3px">\\u2605 ' + d[7] + '</div>' : '';
          // Show current archetype from frame
          let archLabel = '';
          const ci = d[8];
          if (ci != null && curFrame && curFrame.dominant) {{
            const ai = curFrame.dominant[ci];
            if (ai != null && curFrame.archetype_names[ai]) {{
              archLabel = '<br><b>Archetype:</b> ' + curFrame.archetype_names[ai];
            }}
          }}
          // Text keywords
          const textKw = (ci != null && coffees[ci] && coffees[ci].text) ? '<br><b>Notes:</b> ' + coffees[ci].text : '';
          return role + '<b>' + d[2] + '</b><br>Score: ' + d[4] + ' &nbsp; ' + stock + archLabel + flavors + textKw + '<br><span style="color:#666;font-size:10px">Click to open on Sweet Maria\\u2019s</span>';
        }}
      }},
      legend: {{ type: 'plain', top: 5, left: 'center', textStyle: {{ color: '#ccc', fontSize: 11 }}, itemGap: 15 }},
      xAxis: {{ show: false, type: 'value' }},
      yAxis: {{ show: false, type: 'value' }},
      series: series,
      animationDuration: 400,
      animationEasing: 'cubicOut'
    }}, true);
  }}

  renderFrame(0);
  chart.on('click', function(p) {{
    if (p.data && Array.isArray(p.data) && p.data[5]) {{
      window.open(p.data[5], '_blank');
    }}
  }});
  let currentFrameIdx = 0;
  window._renderUmap = function() {{ renderFrame(currentFrameIdx); }};
  const slider = document.getElementById('umap-slider');
  const valEl = document.getElementById('umap-slider-val');
  slider.addEventListener('input', function() {{
    currentFrameIdx = parseInt(this.value);
    valEl.textContent = weights[currentFrameIdx].toFixed(2);
    renderFrame(currentFrameIdx);
  }});
  window.addEventListener('resize', function() {{ chart.resize(); }});
}})();

// --- Section 2: Archetype Radars (split cupping/flavor) ---
(function() {{
  const chartCup = initChart('chart-arch-cupping');
  const chartFlav = initChart('chart-arch-flavor');
  const archs = D.archetypes;
  const names = D.archetype_names;
  const colors = D.archetype_colors;
  const dims = D.dim_names;
  const cupDims = dims.slice(0, 10);
  const flavDims = dims.slice(10);

  function makeRadar(chart, dimNames, dataSlice, maxVal, showLegend) {{
    const indicator = dimNames.map(function(d) {{ return {{ name: d, max: maxVal, min: -maxVal }}; }});
    const series = archs.map(function(a, i) {{
      return {{
        name: names[i], type: 'radar',
        data: [{{ value: dataSlice(a), name: names[i] }}],
        lineStyle: {{ width: 2 }}, areaStyle: {{ opacity: 0.1 }},
        itemStyle: {{ color: colors[i] }}
      }};
    }});
    chart.setOption({{
      tooltip: {{ trigger: 'item' }},
      legend: showLegend ? {{ top: 5, textStyle: {{ color: '#ccc', fontSize: 10 }} }} : {{ show: false }},
      radar: {{ indicator: indicator, shape: 'polygon', splitArea: {{ areaStyle: {{ color: ['#16213e', '#1a2744'] }} }}, axisName: {{ color: '#999', fontSize: 9 }} }},
      series: series
    }});
  }}
  makeRadar(chartCup, cupDims, function(a) {{ return a.slice(0, 10); }}, 3.5, true);
  makeRadar(chartFlav, flavDims, function(a) {{ return a.slice(10); }}, 3.5, false);
  // Sync legend toggles between charts
  let syncing = false;
  chartCup.on('legendselectchanged', function(e) {{
    if (syncing) return;
    syncing = true;
    chartFlav.dispatchAction({{ type: 'legendToggleSelect', name: e.name }});
    syncing = false;
  }});
  window.addEventListener('resize', function() {{ chartCup.resize(); chartFlav.resize(); }});
}})();

// --- Section 3: Catalog Composition ---
(function() {{
  const chart = initChart('chart-composition');
  const pieColors = ['#e94560','#4dc9f6','#50fa7b','#ffb86c','#bd93f9','#ff79c6','#8be9fd','#f1fa8c','#6272a4','#44475a','#282a36','#f8f8f2'];

  function renderPie(data) {{
    chart.setOption({{
      tooltip: {{ trigger: 'item', formatter: '{{b}}: {{c}} coffees ({{d}}%)' }},
      series: [{{
        type: 'pie',
        radius: ['35%', '65%'],
        center: ['50%', '55%'],
        data: data.map(function(d, i) {{ return {{ value: d.count, name: d.name, itemStyle: {{ color: pieColors[i % pieColors.length] }} }}; }}),
        label: {{ color: '#ccc', fontSize: 10 }},
        emphasis: {{ itemStyle: {{ shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' }} }}
      }}]
    }}, true);
  }}

  function getCompData() {{
    const groupBy = document.querySelector('input[name="comp-groupby"]:checked').value;
    const instock = (typeof window._globalStock !== 'undefined') && window._globalStock === 'instock';
    if (groupBy === 'archetype') {{
      const counts = instock ? D.archetype_counts_instock : D.archetype_counts;
      return D.archetype_names.map(function(n, i) {{ return {{ name: n, count: counts[i] }}; }});
    }} else if (groupBy === 'processing') {{
      return instock ? D.composition_processing_instock : D.composition_processing;
    }} else {{
      return instock ? D.composition_region_instock : D.composition_region;
    }}
  }}

  window._renderComposition = function() {{ renderPie(getCompData()); }};
  window._renderComposition();
  document.querySelectorAll('input[name="comp-groupby"]').forEach(function(r) {{
    r.addEventListener('change', function() {{ window._renderComposition(); }});
  }});
  window.addEventListener('resize', function() {{ chart.resize(); }});
}})();

// --- Section 4: Correlations Heatmap ---
(function() {{
  const chart = initChart('chart-correlations');
  const dims = D.dim_names;
  const corr = D.correlations;

  const data = [];
  for (let i = 0; i < dims.length; i++) {{
    for (let j = 0; j < dims.length; j++) {{
      data.push([j, i, corr[i][j]]);
    }}
  }}

  chart.setOption({{
    tooltip: {{
      formatter: function(p) {{
        return dims[p.data[0]] + ' vs ' + dims[p.data[1]] + '<br>r = ' + p.data[2].toFixed(3);
      }}
    }},
    xAxis: {{ type: 'category', data: dims, axisLabel: {{ rotate: 45, fontSize: 9, color: '#999' }} }},
    yAxis: {{ type: 'category', data: dims, axisLabel: {{ fontSize: 9, color: '#999' }} }},
    visualMap: {{
      min: -1, max: 1, calculable: true,
      orient: 'horizontal', left: 'center', bottom: 5,
      inRange: {{ color: ['#e94560', '#1a1a2e', '#4dc9f6'] }},
      textStyle: {{ color: '#ccc' }}
    }},
    series: [{{
      type: 'heatmap',
      data: data,
      emphasis: {{ itemStyle: {{ borderColor: '#fff', borderWidth: 1 }} }}
    }}]
  }});
  window.addEventListener('resize', function() {{ chart.resize(); }});
}})();

// --- Section 5: Box Plots (split cupping 0-10 / flavor 0-5) ---
(function() {{
  const chartCup = initChart('chart-dist-cupping');
  const chartFlav = initChart('chart-dist-flavor');
  const dims = D.dim_names;
  const boxAll = D.box_data;
  const cupDims = dims.slice(0, 10);
  const flavDims = dims.slice(10);
  const cupBox = boxAll.slice(0, 10);
  const flavBox = boxAll.slice(10);

  function renderAll() {{
    chartCup.setOption({{
      title: {{ text: 'Cupping Scores (0\\u201310)', textStyle: {{ color: '#ccc', fontSize: 12 }}, left: 'center' }},
      tooltip: {{ trigger: 'item' }},
      xAxis: {{ type: 'category', data: cupDims, axisLabel: {{ rotate: 45, fontSize: 9, color: '#999' }} }},
      yAxis: {{ type: 'value', min: 0, max: 10, name: 'Score', nameTextStyle: {{ color: '#999' }}, axisLabel: {{ color: '#999' }} }},
      series: [{{ type: 'boxplot', data: cupBox, itemStyle: {{ color: '#4dc9f6', borderColor: '#4dc9f6' }} }}],
      animation: false
    }}, true);
    chartFlav.setOption({{
      title: {{ text: 'Flavor Intensity (0\\u20135)', textStyle: {{ color: '#ccc', fontSize: 12 }}, left: 'center' }},
      tooltip: {{ trigger: 'item' }},
      xAxis: {{ type: 'category', data: flavDims, axisLabel: {{ rotate: 45, fontSize: 9, color: '#999' }} }},
      yAxis: {{ type: 'value', min: 0, max: 5, name: 'Score', nameTextStyle: {{ color: '#999' }}, axisLabel: {{ color: '#999' }} }},
      series: [{{ type: 'boxplot', data: flavBox, itemStyle: {{ color: '#50fa7b', borderColor: '#50fa7b' }} }}],
      animation: false
    }}, true);
  }}

  function renderArch() {{
    const groups = D.box_archetype_groups;
    const names = D.archetype_names;
    const colors = D.archetype_colors;
    const cupSeries = names.map(function(n, ai) {{
      return {{ name: n, type: 'boxplot', data: groups[ai].slice(0, 10), itemStyle: {{ color: colors[ai], borderColor: colors[ai] }} }};
    }});
    const flavSeries = names.map(function(n, ai) {{
      return {{ name: n, type: 'boxplot', data: groups[ai].slice(10), itemStyle: {{ color: colors[ai], borderColor: colors[ai] }} }};
    }});
    chartCup.setOption({{
      title: {{ text: 'Cupping Scores (0\\u201310)', textStyle: {{ color: '#ccc', fontSize: 12 }}, left: 'center' }},
      tooltip: {{ trigger: 'item' }},
      legend: {{ top: 20, textStyle: {{ color: '#ccc' }} }},
      xAxis: {{ type: 'category', data: cupDims, axisLabel: {{ rotate: 45, fontSize: 9, color: '#999' }} }},
      yAxis: {{ type: 'value', min: 0, max: 10, name: 'Score', nameTextStyle: {{ color: '#999' }}, axisLabel: {{ color: '#999' }} }},
      series: cupSeries,
      animation: false
    }}, true);
    chartFlav.setOption({{
      title: {{ text: 'Flavor Intensity (0\\u20135)', textStyle: {{ color: '#ccc', fontSize: 12 }}, left: 'center' }},
      tooltip: {{ trigger: 'item' }},
      legend: {{ show: false }},
      xAxis: {{ type: 'category', data: flavDims, axisLabel: {{ rotate: 45, fontSize: 9, color: '#999' }} }},
      yAxis: {{ type: 'value', min: 0, max: 5, name: 'Score', nameTextStyle: {{ color: '#999' }}, axisLabel: {{ color: '#999' }} }},
      series: flavSeries,
      animation: false
    }}, true);
    // Sync legend between box plot charts
    chartCup.off('legendselectchanged');
    var boxSyncing = false;
    chartCup.on('legendselectchanged', function(e) {{
      if (boxSyncing) return;
      boxSyncing = true;
      chartFlav.dispatchAction({{ type: 'legendToggleSelect', name: e.name }});
      boxSyncing = false;
    }});
  }}

  renderAll();
  function refresh() {{
    const mode = document.querySelector('input[name="box-group"]:checked').value;
    if (mode === 'archetype') renderArch();
    else renderAll();
  }}
  window._renderBoxPlots = refresh;
  document.querySelectorAll('input[name="box-group"]').forEach(function(r) {{
    r.addEventListener('change', refresh);
  }});
  window.addEventListener('resize', function() {{ chartCup.resize(); chartFlav.resize(); }});
}})();

// --- Section 6: PCA ---
(function() {{
  const chartVar = initChart('chart-pca-var');
  const chartLoad = initChart('chart-pca-loadings');
  const variance = D.pca_variance;
  const loadings = D.pca_loadings;
  const dims = D.dim_names;
  const nKept = D.pca_n_kept;

  // Variance bar chart
  let cumul = 0;
  const cumulData = variance.map(function(v) {{ cumul += v; return cumul; }});
  chartVar.setOption({{
    tooltip: {{ trigger: 'axis' }},
    xAxis: {{ type: 'category', data: variance.map(function(_, i) {{ return 'F' + (i+1); }}), axisLabel: {{ color: '#999' }} }},
    yAxis: {{ type: 'value', max: 100, name: '% variance', nameTextStyle: {{ color: '#999' }}, axisLabel: {{ color: '#999' }} }},
    series: [
      {{ name: 'Individual', type: 'bar', data: variance.map(function(v) {{ return (v*100).toFixed(1); }}), itemStyle: {{ color: '#4dc9f6' }}, markLine: {{ silent: true, data: [{{ yAxis: 80, lineStyle: {{ color: '#e94560', type: 'dashed' }} }}] }} }},
      {{ name: 'Cumulative', type: 'line', data: cumulData.map(function(v) {{ return (v*100).toFixed(1); }}), itemStyle: {{ color: '#e94560' }} }}
    ],
    legend: {{ textStyle: {{ color: '#ccc' }} }}
  }});

  // Loadings heatmap
  const loadData = [];
  for (let fi = 0; fi < loadings.length; fi++) {{
    for (let di = 0; di < dims.length; di++) {{
      loadData.push([di, fi, loadings[fi][di]]);
    }}
  }}
  chartLoad.setOption({{
    tooltip: {{ formatter: function(p) {{ return 'F' + (p.data[1]+1) + ' × ' + dims[p.data[0]] + '<br>Loading: ' + p.data[2].toFixed(3); }} }},
    xAxis: {{ type: 'category', data: dims, axisLabel: {{ rotate: 45, fontSize: 9, color: '#999' }} }},
    yAxis: {{ type: 'category', data: loadings.map(function(_, i) {{ return 'Factor ' + (i+1); }}), axisLabel: {{ color: '#999' }} }},
    visualMap: {{
      min: -0.6, max: 0.6, calculable: true,
      orient: 'horizontal', left: 'center', bottom: 5,
      inRange: {{ color: ['#e94560', '#1a1a2e', '#4dc9f6'] }},
      textStyle: {{ color: '#ccc' }}
    }},
    series: [{{ type: 'heatmap', data: loadData, emphasis: {{ itemStyle: {{ borderColor: '#fff', borderWidth: 1 }} }} }}]
  }});
  window.addEventListener('resize', function() {{ chartVar.resize(); chartLoad.resize(); }});
}})();

// --- Section 7: Processing Methods ---
(function() {{
  const chartCup = initChart('chart-proc-cupping');
  const chartFlav = initChart('chart-proc-flavor');
  const methods = D.processing_methods;
  const dims = D.dim_names;
  const cupDims = dims.slice(0, 10);
  const flavDims = dims.slice(10);
  const colors = ['#e94560','#4dc9f6','#50fa7b','#ffb86c','#bd93f9','#ff79c6','#8be9fd','#f1fa8c'];

  // Compute dynamic range
  let maxAbsCup = 0, maxAbsFlav = 0;
  methods.slice(0, 8).forEach(function(m) {{
    m.values.slice(0, 10).forEach(function(v) {{ if (Math.abs(v) > maxAbsCup) maxAbsCup = Math.abs(v); }});
    m.values.slice(10).forEach(function(v) {{ if (Math.abs(v) > maxAbsFlav) maxAbsFlav = Math.abs(v); }});
  }});
  const maxCup = Math.ceil(maxAbsCup * 1.2 * 10) / 10;
  const maxFlav = Math.ceil(maxAbsFlav * 1.2 * 10) / 10;

  function makeRadar(chart, dimNames, sliceFn, maxVal, showLegend) {{
    const indicator = dimNames.map(function(d) {{ return {{ name: d, max: maxVal, min: -maxVal }}; }});
    const series = methods.slice(0, 8).map(function(m, i) {{
      return {{
        name: m.name + ' (' + m.count + ')',
        type: 'radar',
        data: [{{ value: sliceFn(m.values), name: m.name }}],
        lineStyle: {{ width: 2 }}, areaStyle: {{ opacity: 0.05 }},
        itemStyle: {{ color: colors[i % colors.length] }}
      }};
    }});
    chart.setOption({{
      tooltip: {{ trigger: 'item' }},
      legend: showLegend ? {{ type: 'plain', top: 5, left: 'center', textStyle: {{ color: '#ccc', fontSize: 11 }}, itemGap: 15 }} : {{ show: false }},
      radar: {{ indicator: indicator, shape: 'polygon', radius: '55%', splitArea: {{ areaStyle: {{ color: ['#16213e', '#1a2744'] }} }}, axisName: {{ color: '#999', fontSize: 9 }} }},
      series: series
    }});
  }}
  makeRadar(chartCup, cupDims, function(v) {{ return v.slice(0, 10); }}, maxCup, true);
  makeRadar(chartFlav, flavDims, function(v) {{ return v.slice(10); }}, maxFlav, false);
  // Sync legend toggles between charts
  var procSyncing = false;
  chartCup.on('legendselectchanged', function(e) {{
    if (procSyncing) return;
    procSyncing = true;
    chartFlav.dispatchAction({{ type: 'legendToggleSelect', name: e.name }});
    procSyncing = false;
  }});
  window.addEventListener('resize', function() {{ chartCup.resize(); chartFlav.resize(); }});
}})();

// --- Section 8: Explore (Dimension Contrast Pairs) ---
(function() {{
  const chart = initChart('chart-explore');
  const pairsInstock = D.explore_pairs;
  const pairsAll = D.explore_pairs_all;

  let currentPairs = pairsInstock;

  function renderExplore(pairs) {{
    currentPairs = pairs;
    const dims = pairs.map(function(p) {{ return p.dim; }});
    const highVals = pairs.map(function(p) {{ return p.high_val; }});
    const lowVals = pairs.map(function(p) {{ return p.low_val; }});

    chart.setOption({{
      tooltip: {{
        trigger: 'axis',
        axisPointer: {{ type: 'shadow' }},
        formatter: function(params) {{
          const idx = params[0].dataIndex;
          const p = pairs[idx];
          return '<b>' + p.dim + '</b><br>' +
            '\\u25b2 HIGH: ' + p.high_name + ' (' + p.high_val.toFixed(1) + ')<br>' +
            '\\u25bc LOW: ' + p.low_name + ' (' + p.low_val.toFixed(1) + ')<br>' +
            'Pair similarity: ' + p.similarity.toFixed(2) +
            '<br><span style="color:#666;font-size:10px">Click a dot to open on Sweet Maria\\u2019s</span>';
        }}
      }},
      xAxis: {{ type: 'value', name: 'Score', nameTextStyle: {{ color: '#999' }}, axisLabel: {{ color: '#999' }} }},
      yAxis: {{ type: 'category', data: dims, axisLabel: {{ color: '#999', fontSize: 10 }}, inverse: true }},
      series: [
        {{ name: 'HIGH', type: 'scatter', data: highVals.map(function(v, i) {{ return [v, i]; }}), symbolSize: 12, itemStyle: {{ color: '#50fa7b' }} }},
        {{ name: 'LOW', type: 'scatter', data: lowVals.map(function(v, i) {{ return [v, i]; }}), symbolSize: 12, itemStyle: {{ color: '#e94560' }} }},
      ],
      legend: {{ textStyle: {{ color: '#ccc' }} }}
    }}, true);
  }}

  renderExplore(pairsInstock);
  chart.on('click', function(p) {{
    if (!p || p.dataIndex == null) return;
    const pair = currentPairs[p.dataIndex];
    if (!pair) return;
    const url = p.seriesName === 'HIGH' ? pair.high_url : pair.low_url;
    if (url) window.open(url, '_blank');
  }});
  window._renderExplore = function() {{
    const instock = (typeof window._globalStock !== 'undefined') && window._globalStock === 'instock';
    renderExplore(instock ? pairsInstock : pairsAll);
  }};
  window.addEventListener('resize', function() {{ chart.resize(); }});
}})();

// --- Section 9: Superlatives Table ---
(function() {{
  const tbody = document.getElementById('superlatives-body');

  function coffeeLink(name, url) {{
    return '<a href="' + url + '" target="_blank" style="color:#4dc9f6">' + name + '</a>';
  }}

  function renderTable(superlatives, outliers, typicals) {{
    let html = '';
    superlatives.forEach(function(s) {{
      html += '<tr><td>' + s.category + '</td><td>' + coffeeLink(s.name, s.url) + '</td><td>' + s.value + '</td></tr>';
    }});
    html += '<tr><td colspan="3" style="padding-top:1rem;color:#e94560;font-weight:bold">Most Unusual (farthest from average)</td></tr>';
    outliers.forEach(function(o) {{
      html += '<tr><td>Outlier (dist=' + o.dist.toFixed(2) + ')</td><td>' + coffeeLink(o.name, o.url) + '</td><td>' + o.dims + '</td></tr>';
    }});
    html += '<tr><td colspan="3" style="padding-top:1rem;color:#4dc9f6;font-weight:bold">Most Typical (closest to average)</td></tr>';
    typicals.forEach(function(t) {{
      html += '<tr><td>Typical (dist=' + t.dist.toFixed(2) + ')</td><td>' + coffeeLink(t.name, t.url) + '</td><td></td></tr>';
    }});
    tbody.innerHTML = html;
  }}

  renderTable(D.superlatives, D.outliers, D.typicals);
  window._renderSuperlatives = function() {{
    const instock = (typeof window._globalStock !== 'undefined') && window._globalStock === 'instock';
    if (instock) {{
      renderTable(D.superlatives_instock, D.outliers_instock, D.typicals_instock);
    }} else {{
      renderTable(D.superlatives, D.outliers, D.typicals);
    }}
  }};
}})();

// --- Global In-Stock Toggle ---
(function() {{
  window._globalStock = 'all';
  const countLabel = document.getElementById('stock-count-label');
  document.querySelectorAll('input[name="global-stock"]').forEach(function(r) {{
    r.addEventListener('change', function() {{
      window._globalStock = this.value;
      countLabel.textContent = this.value === 'instock'
        ? D.n_instock + ' in stock'
        : D.n_coffees + ' coffees';
      // Refresh all affected sections
      if (window._renderUmap) window._renderUmap();
      if (window._renderComposition) window._renderComposition();
      if (window._renderBoxPlots) window._renderBoxPlots();
      if (window._renderExplore) window._renderExplore();
      if (window._renderSuperlatives) window._renderSuperlatives();
    }});
  }});
}})();

</script>
</body>
</html>"""
