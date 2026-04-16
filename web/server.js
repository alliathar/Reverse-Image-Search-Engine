const express = require('express');
const multer = require('multer');
const { execFile } = require('child_process');
const path = require('path');
const cors = require('cors');
const fs = require('fs');

const app = express();
const port = 3000;

app.use(cors());
app.use(express.static('public')); // Serve the frontend

// Multer config for query image upload
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir);
}
const upload = multer({ dest: 'uploads/' });

// API endpoint for search
app.post('/api/search', upload.single('queryImage'), (req, res) => {
    if (!req.file || !req.body.datasetPath) {
        return res.status(400).json({ error: 'Missing query image or dataset path.' });
    }

    const queryImagePath = req.file.path;
    const datasetPath = req.body.datasetPath;

    // Path to compiled C++ Executable
    const exePath = path.join(__dirname, '..', 'build', 'ReverseImageSearch.exe');
    
    // We allow up to 50MB of buffer because the graph JSON could be large for large datasets
    execFile(exePath, ['search', queryImagePath, datasetPath], { maxBuffer: 1024 * 1024 * 50 }, (error, stdout, stderr) => {
        // Clean up the uploaded file
        fs.unlink(queryImagePath, () => {});

        if (error) {
            console.error('Execution error:', error);
            console.error('stderr:', stderr);
            return res.status(500).json({ error: 'Failed to process image search.' });
        }

        try {
            const jsonStartOffset = stdout.indexOf('{');
            if (jsonStartOffset === -1) throw new Error('No JSON output found from engine.');
            const cleanJsonStr = stdout.substring(jsonStartOffset);
            
            const results = JSON.parse(cleanJsonStr);
            res.json(results);
        } catch (e) {
            console.error('JSON parsing error:', e);
            console.log('Raw output:', stdout);
            res.status(500).json({ error: 'Failed to parse engine output.' });
        }
    });
});

// API endpoint to serve arbitrary local target images to the frontend securely
app.get('/api/image', (req, res) => {
    const targetPath = req.query.path;
    if (!targetPath) {
        return res.status(400).send('No path provided');
    }
    res.sendFile(path.resolve(targetPath), (err) => {
        if (err) {
            res.status(404).send('Image not found');
        }
    });
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
