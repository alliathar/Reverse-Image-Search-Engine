const searchForm = document.getElementById('search-form');
const queryImageInput = document.getElementById('queryImage');
const fileNameDisplay = document.getElementById('file-name-display');
const submitBtn = document.getElementById('submit-btn');
const btnText = document.querySelector('.btn-text');
const loader = document.querySelector('.loader');
const resultsContainer = document.getElementById('results-container');
const matchesGrid = document.getElementById('matches-grid');
const loadBtn = document.getElementById('load-btn');
const datasetPathInput = document.getElementById('datasetPath');
const loadStatus = document.getElementById('load-status');
const queryPreview = document.getElementById('query-preview');
const queryThumbs = document.getElementById('query-thumbs');
const searchModeSelect = document.getElementById('search-mode');
const categoryGroup = document.getElementById('category-group');
const categorySelect = document.getElementById('category');
const uploadLabelText = document.getElementById('upload-label-text');
const graphSection = document.getElementById('graph-section');
const graphToggle = document.getElementById('graph-toggle');
let lastQueryUrls = [];
let cachedGraph = null;
let graphRendered = false;

// ----- Folder picker -----
const browseBtn = document.getElementById('browse-btn');
const browseModal = document.getElementById('browse-modal');
const browseClose = document.getElementById('browse-close');
const browseCancel = document.getElementById('browse-cancel');
const browseSelect = document.getElementById('browse-select');
const browseUp = document.getElementById('browse-up');
const browseCurrent = document.getElementById('browse-current');
const browseList = document.getElementById('browse-list');
const browseFilter = document.getElementById('browse-filter');
let currentFolders = [];
let browseCurrentPath = null;
let browseParentPath = null;

async function loadFolder(targetPath) {
    browseList.innerHTML = '<li class="folder-empty">Loading…</li>';
    try {
        const url = targetPath
            ? `/api/browse?path=${encodeURIComponent(targetPath)}`
            : '/api/browse';
        const res = await fetch(url);
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        browseCurrentPath = data.path;
        browseParentPath = data.parent;
        browseCurrent.textContent = data.path;
        browseUp.disabled = !data.parent;

        currentFolders = data.folders;
        browseFilter.value = '';
        renderBrowseList();
        browseFilter.focus();
    } catch (e) {
        browseList.innerHTML = `<li class="folder-empty">Error: ${e.message}</li>`;
    }
}

function renderBrowseList() {
    const q = browseFilter.value.trim().toLowerCase();
    // Score each folder. Exact prefix wins, then substring, then a loose
    // "matches the typed letters in order" check so "dl" finds "Downloads".
    const scored = currentFolders
        .map(name => {
            const lower = name.toLowerCase();
            if (!q) return { name, score: 0 };
            if (lower.startsWith(q)) return { name, score: 1 };
            if (lower.includes(q)) return { name, score: 2 };
            // Subsequence: every letter of q appears in order in lower.
            let i = 0;
            for (const c of lower) { if (c === q[i]) i++; if (i === q.length) break; }
            if (i === q.length) return { name, score: 3 };
            return null;
        })
        .filter(Boolean)
        .sort((a, b) => a.score - b.score || a.name.localeCompare(b.name));

    if (scored.length === 0) {
        browseList.innerHTML = '<li class="folder-empty">No matches</li>';
        return;
    }
    browseList.innerHTML = '';
    for (const { name } of scored) {
        const li = document.createElement('li');
        li.className = 'folder-item';
        li.innerHTML = `<span class="folder-icon">📁</span> ${name}`;
        li.addEventListener('click', () => {
            const sep = browseCurrentPath.endsWith('/') || browseCurrentPath.endsWith('\\') ? '' : '/';
            loadFolder(browseCurrentPath + sep + name);
        });
        browseList.appendChild(li);
    }
}

function openBrowse() {
    browseModal.classList.remove('hidden');
    loadFolder(datasetPathInput.value || null);
}
function closeBrowse() {
    browseModal.classList.add('hidden');
}

browseBtn.addEventListener('click', openBrowse);
browseClose.addEventListener('click', closeBrowse);
browseCancel.addEventListener('click', closeBrowse);
browseModal.addEventListener('click', (e) => { if (e.target === browseModal) closeBrowse(); });
browseUp.addEventListener('click', () => { if (browseParentPath) loadFolder(browseParentPath); });
browseSelect.addEventListener('click', () => {
    if (browseCurrentPath) {
        datasetPathInput.value = browseCurrentPath;
        loadBtn.disabled = false;
        closeBrowse();
    }
});

browseFilter.addEventListener('input', renderBrowseList);
browseFilter.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        // Open the first visible folder.
        e.preventDefault();
        const first = browseList.querySelector('.folder-item');
        if (first) first.click();
    } else if (e.key === 'Escape') {
        if (browseFilter.value) {
            browseFilter.value = '';
            renderBrowseList();
        } else {
            closeBrowse();
        }
    }
});

queryImageInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) {
        fileNameDisplay.textContent = 'No file selected';
    } else if (files.length === 1) {
        fileNameDisplay.textContent = files[0].name;
    } else {
        fileNameDisplay.textContent = `${files.length} files selected`;
    }
});

// Mode dropdown toggles category visibility and multi-file selection.
function applyMode() {
    const mode = searchModeSelect.value;
    const multi = mode === 'multi';
    const needsCategory = mode === 'category';

    queryImageInput.multiple = multi;
    uploadLabelText.textContent = multi ? 'Select Query Images' : 'Select Query Image';
    categoryGroup.classList.toggle('hidden', !needsCategory);
    categorySelect.required = needsCategory;
}
searchModeSelect.addEventListener('change', applyMode);
applyMode();

let networkInstance = null;

searchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const mode = searchModeSelect.value;
    const files = Array.from(queryImageInput.files);

    if (files.length === 0) return;
    if (mode === 'category' && !categorySelect.value) {
        alert('Pick a category for category-wise search.');
        return;
    }

    // Build the request payload. The backend expects `mode` to be normal/negative/multi;
    // "category" is a UI label for normal search with a category filter.
    const formData = new FormData();
    formData.append('mode', mode === 'category' ? 'normal' : mode);
    formData.append('category', (mode === 'category') ? categorySelect.value : 'all');
    const filesToSend = (mode === 'multi') ? files : [files[0]];
    filesToSend.forEach(f => formData.append('queryImage', f));

    // Render query thumbnails (revoke any previous object URLs to avoid leaks).
    lastQueryUrls.forEach(u => URL.revokeObjectURL(u));
    lastQueryUrls = filesToSend.map(f => URL.createObjectURL(f));
    queryThumbs.innerHTML = '';
    lastQueryUrls.forEach(url => {
        const img = document.createElement('img');
        img.src = url;
        img.alt = 'Query';
        queryThumbs.appendChild(img);
    });
    queryPreview.classList.remove('hidden');

    // UI Loading state
    submitBtn.disabled = true;
    btnText.classList.add('hidden');
    loader.classList.remove('hidden');
    resultsContainer.classList.add('hidden');
    
    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || 'Server error');
        }
        
        const data = await response.json();
        renderResults(data, mode);
    } catch (error) {
        console.error("Search failed:", error);
        alert(`Search failed: ${error.message}`);
    } finally {
        submitBtn.disabled = false;
        btnText.classList.remove('hidden');
        loader.classList.add('hidden');
    }
});

loadBtn.addEventListener('click', async () => {
    const datasetPath = datasetPathInput.value.trim();
    if (!datasetPath) return;

    loadBtn.disabled = true;
    loadStatus.textContent = 'Loading…';

    try {
        const res = await fetch('/api/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ datasetPath })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        loadStatus.textContent = `Ready — ${data.count} images loaded`;
        submitBtn.disabled = false;

        // Populate the category dropdown.
        const cats = Array.isArray(data.categories) ? data.categories : [];
        categorySelect.innerHTML = '<option value="">Pick a category…</option>'
            + cats.map(c => `<option value="${c}">${c}</option>`).join('');
        categorySelect.disabled = cats.length === 0;

        // Cache the graph; user opens it explicitly with the toggle.
        cachedGraph = data.graph || null;
        graphRendered = false;
        graphSection.classList.add('hidden');
        resultsContainer.classList.add('graph-hidden');
        graphToggle.textContent = 'Show neural map';
    } catch (e) {
        loadStatus.textContent = `Error: ${e.message}`;
    }
    loadBtn.disabled = false;
});

graphToggle.addEventListener('click', () => {
    const isHidden = graphSection.classList.contains('hidden');
    if (isHidden) {
        graphSection.classList.remove('hidden');
        resultsContainer.classList.remove('graph-hidden');
        graphToggle.textContent = 'Hide neural map';
        // Render the graph lazily, only the first time it's opened.
        if (!graphRendered && cachedGraph) {
            setTimeout(() => renderGraph(cachedGraph), 0);
            graphRendered = true;
        }
    } else {
        graphSection.classList.add('hidden');
        resultsContainer.classList.add('graph-hidden');
        graphToggle.textContent = 'Show neural map';
    }
});

function renderResults(data, mode) {
    matchesGrid.innerHTML = '';
    
    // Filter out null/invalid items if any
    const validResults = (data.results || []).filter(r => r && r.path);
    
    const uniquenessWidget = document.getElementById('uniqueness-widget');
    const uniquenessValue = document.getElementById('uniqueness-value');
    const gaugeFill = document.getElementById('gauge-fill');
    const uniquenessDesc = document.getElementById('uniqueness-desc');

    if (validResults.length > 0) {
        // Show uniqueness widget ONLY for normal or category searches
        if (mode === 'normal' || mode === 'category') {
            uniquenessWidget.classList.remove('hidden');
            const bestMatchAccuracy = validResults[0].accuracy || 0;
            const uniqueness = Math.max(0, 100 - bestMatchAccuracy);
            
            uniquenessValue.textContent = `${uniqueness.toFixed(1)}%`;
            const strokeOffset = 126 - (126 * uniqueness / 100);
            gaugeFill.style.strokeDashoffset = strokeOffset;
            
            if (uniqueness > 70) {
                gaugeFill.style.stroke = '#10b981'; // Green
                uniquenessDesc.textContent = 'Highly unique. No similar images found in dataset.';
            } else if (uniqueness > 30) {
                gaugeFill.style.stroke = '#f59e0b'; // Yellow
                uniquenessDesc.textContent = 'Somewhat unique. Has some similar features in dataset.';
            } else {
                gaugeFill.style.stroke = '#ef4444'; // Red
                uniquenessDesc.textContent = 'Not unique. Very similar images exist in dataset.';
            }
        } else {
            uniquenessWidget.classList.add('hidden');
        }

        validResults.forEach((match, index) => {
            const card = document.createElement('div');
            card.className = 'match-card';
            card.style.animationDelay = `${index * 0.1}s`;
            
            // Construct the path to trigger /api/image safely
            const encodedPath = encodeURIComponent(match.path);
            const imgSrc = `/api/image?path=${encodedPath}`;
            
            // Extract filename safely (handling both forward and backslashes)
            const fileName = match.path.split(/[/\\]/).pop();
            
            // Color based on accuracy (green/blue scale)
            // high accuracy -> hue ~ 190 (cyan/blue), lower -> hue ~ 280 (purple)
            const hue = Math.max(150, match.accuracy * 2.5); // Just a nice gradient math
            const color = `hsl(${hue}, 80%, 60%)`;
            
            let luminanceMatch = false;
            let broadShapes = { total: 0, matched: 0 };
            let horizontal = { total: 0, matched: 0 };
            let vertical = { total: 0, matched: 0 };
            let textures = { total: 0, matched: 0 };

            if (data.queryHash && match.hash) {
                luminanceMatch = data.queryHash[0] === match.hash[0];
                for (let u = 0; u < 8; u++) {
                    for (let v = 0; v < 8; v++) {
                        if (u === 0 && v === 0) continue;
                        
                        let bitIndex = u * 8 + v;
                        const isMatch = data.queryHash[bitIndex] === match.hash[bitIndex];
                        
                        if (u + v <= 3) {
                            broadShapes.total++;
                            if (isMatch) broadShapes.matched++;
                        } else if (u <= 1 && v >= 4) {
                            horizontal.total++;
                            if (isMatch) horizontal.matched++;
                        } else if (u >= 4 && v <= 1) {
                            vertical.total++;
                            if (isMatch) vertical.matched++;
                        } else if (u + v >= 9) {
                            textures.total++;
                            if (isMatch) textures.matched++;
                        }
                    }
                }
            }

            const getPercentage = (cat) => cat.total > 0 ? (cat.matched / cat.total) * 100 : 0;
            
            const renderFeatureRow = (icon, name, pct) => {
                const color = `hsl(${Math.max(10, pct * 1.5)}, 80%, 50%)`;
                return `
                    <div class="feature-row">
                        <div class="feature-info">
                            <span class="feature-icon">${icon}</span>
                            <span class="feature-name">${name}</span>
                        </div>
                        <div class="feature-bar-container">
                            <div class="feature-bar">
                                <div class="feature-fill" style="width: ${pct}%; background: ${color}"></div>
                            </div>
                            <span class="feature-pct">${Math.round(pct)}%</span>
                        </div>
                    </div>
                `;
            };

            const featureAnalysisHTML = `
                <div class="feature-analysis-list">
                    <div class="feature-row">
                        <div class="feature-info">
                            <span class="feature-icon">☀️</span>
                            <span class="feature-name">Luminance (Brightness)</span>
                        </div>
                        <div class="feature-status ${luminanceMatch ? 'match' : 'mismatch'}">
                            ${luminanceMatch ? 'Match' : 'Mismatch'}
                        </div>
                    </div>
                    ${renderFeatureRow('🏔️', 'Broad Shapes (Low Freq)', getPercentage(broadShapes))}
                    ${renderFeatureRow('⬌', 'Horizontal Structure', getPercentage(horizontal))}
                    ${renderFeatureRow('⬍', 'Vertical Structure', getPercentage(vertical))}
                    ${renderFeatureRow('✨', 'Fine Textures & Edges', getPercentage(textures))}
                </div>
            `;

            card.innerHTML = `
                <div class="expand-btn" title="View Semantic Feature Analysis">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>
                </div>
                <div class="match-image-container">
                    <img src="${imgSrc}" class="match-img" alt="Match" onerror="this.src='data:image/svg+xml;utf8,<svg viewBox=\\'0 0 100 100\\' xmlns=\\'http://www.w3.org/2000/svg\\'><rect width=\\'100\\' height=\\'100\\' fill=\\'#333\\'/><text x=\\'50\\' y=\\'50\\' fill=\\'#777\\' text-anchor=\\'middle\\' alignment-baseline=\\'middle\\'>Error loading</text></svg>'">
                </div>
                <div class="match-details">
                    <div class="match-path" title="${match.path}">${fileName}</div>
                    <div style="font-size: 0.75rem; color: var(--primary); margin-bottom: 0.5rem; display: inline-block; padding: 2px 6px; background: rgba(139, 92, 246, 0.1); border-radius: 4px; font-weight: 600; text-transform: capitalize;">${match.category || 'uncategorized'}</div>
                    <div class="accuracy-gauge">
                        <div class="accuracy-bar">
                            <div class="accuracy-fill" style="width: 0%; background: ${color}"></div>
                        </div>
                        <span class="accuracy-text" style="color: ${color}">${match.accuracy.toFixed(1)}%</span>
                    </div>
                </div>
                <div class="metric-details">
                    <div class="metric-header">
                        <span>Feature Match Analysis</span>
                    </div>
                    ${featureAnalysisHTML}
                </div>
            `;

            matchesGrid.appendChild(card);
            
            // Toggle metric details
            const expandBtn = card.querySelector('.expand-btn');
            const metricDetails = card.querySelector('.metric-details');
            if (expandBtn && metricDetails) {
                expandBtn.addEventListener('click', () => {
                    expandBtn.classList.toggle('active');
                    metricDetails.classList.toggle('expanded');
                });
            }
            
            // Trigger animation slightly delayed
            setTimeout(() => {
                card.querySelector('.accuracy-fill').style.width = `${Math.min(100, match.accuracy)}%`;
            }, 100 + index * 50);
        });
    } else {
        if (uniquenessWidget) uniquenessWidget.classList.add('hidden');
        matchesGrid.innerHTML = '<p style="color: var(--text-secondary)">No matches were found.</p>';
    }
    
    resultsContainer.classList.remove('hidden');
}

function renderGraph(graphData) {
    const container = document.getElementById('network-canvas');
    
    // Safety check
    if (!graphData.nodes || !graphData.edges) {
        container.innerHTML = '<p style="padding: 1rem; color: #999;">Graph data unavailable</p>';
        return;
    }
    
    // Cap nodes to keep rendering fast
    const MAX_NODES = 150;
    const visibleNodes = graphData.nodes.slice(0, MAX_NODES);
    const visibleIds = new Set(visibleNodes.map(n => n.id));

    // Map graph nodes to vis.js format
    const nodesDataSet = new vis.DataSet(visibleNodes.map(n => {
        let labelName = `ID:${n.id}`;
        if (n.path && n.path !== 'unknown') {
            labelName = n.path.split(/[/\\]/).pop().substring(0, 10);
        }
        return {
            id: n.id,
            label: labelName,
            title: n.path,
            group: n.layer,
            value: n.layer + 1,
            shape: 'dot'
        };
    }));

    // Map edges (only between visible nodes)
    const edgesDataSet = new vis.DataSet(graphData.edges
        .filter(e => visibleIds.has(e.source) && visibleIds.has(e.target))
        .map(e => ({
            from: e.source,
            to: e.target,
            color: { color: 'rgba(255,255,255,0.15)', highlight: '#ec4899' }
        })));

    const data = { nodes: nodesDataSet, edges: edgesDataSet };

    const options = {
        nodes: {
            font: { color: '#f8fafc', size: 12, face: 'Inter' },
            color: { background: '#8b5cf6', border: 'transparent', highlight: { background: '#ec4899', border: '#fff' } }
        },
        physics: {
            forceAtlas2Based: { gravitationalConstant: -80, centralGravity: 0.01, springLength: 80, springConstant: 0.08 },
            solver: 'forceAtlas2Based',
            stabilization: { enabled: false }
        },
        interaction: { hover: true, tooltipDelay: 200 }
    };
    
    if (networkInstance) {
        networkInstance.destroy();
    }
    networkInstance = new vis.Network(container, data, options);
}
