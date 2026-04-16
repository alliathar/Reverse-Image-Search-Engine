const searchForm = document.getElementById('search-form');
const queryImageInput = document.getElementById('queryImage');
const fileNameDisplay = document.getElementById('file-name-display');
const submitBtn = document.getElementById('submit-btn');
const btnText = document.querySelector('.btn-text');
const loader = document.querySelector('.loader');
const resultsContainer = document.getElementById('results-container');
const matchesGrid = document.getElementById('matches-grid');

queryImageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        fileNameDisplay.textContent = e.target.files[0].name;
    } else {
        fileNameDisplay.textContent = "No file selected";
    }
});

let networkInstance = null;

searchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(searchForm);
    
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
        renderResults(data);
    } catch (error) {
        console.error("Search failed:", error);
        alert(`Search failed: ${error.message}`);
    } finally {
        submitBtn.disabled = false;
        btnText.classList.remove('hidden');
        loader.classList.add('hidden');
    }
});

function renderResults(data) {
    matchesGrid.innerHTML = '';
    
    // Filter out null/invalid items if any
    const validResults = (data.results || []).filter(r => r && r.path);
    
    if (validResults.length > 0) {
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
            
            card.innerHTML = `
                <div class="match-image-container">
                    <img src="${imgSrc}" class="match-img" alt="Match" onerror="this.src='data:image/svg+xml;utf8,<svg viewBox=\\'0 0 100 100\\' xmlns=\\'http://www.w3.org/2000/svg\\'><rect width=\\'100\\' height=\\'100\\' fill=\\'#333\\'/><text x=\\'50\\' y=\\'50\\' fill=\\'#777\\' text-anchor=\\'middle\\' alignment-baseline=\\'middle\\'>Error loading</text></svg>'">
                </div>
                <div class="match-details">
                    <div class="match-path" title="${match.path}">${fileName}</div>
                    <div class="accuracy-gauge">
                        <div class="accuracy-bar">
                            <div class="accuracy-fill" style="width: 0%; background: ${color}"></div>
                        </div>
                        <span class="accuracy-text" style="color: ${color}">${match.accuracy.toFixed(1)}%</span>
                    </div>
                </div>
            `;
            
            matchesGrid.appendChild(card);
            
            // Trigger animation slightly delayed
            setTimeout(() => {
                card.querySelector('.accuracy-fill').style.width = `${Math.min(100, match.accuracy)}%`;
            }, 100 + index * 50);
        });
    } else {
        matchesGrid.innerHTML = '<p style="color: var(--text-secondary)">No matches were found.</p>';
    }
    
    // Render Graph
    if (data.graph) {
        renderGraph(data.graph);
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
    
    // Map graph nodes to vis.js format
    const nodesDataSet = new vis.DataSet(graphData.nodes.map(n => {
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
    
    // Map edges
    const edgesDataSet = new vis.DataSet(graphData.edges.map(e => ({
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
            stabilization: { iterations: 150 }
        },
        interaction: { hover: true, tooltipDelay: 200 }
    };
    
    if (networkInstance) {
        networkInstance.destroy();
    }
    networkInstance = new vis.Network(container, data, options);
}
