import React, { useState, useRef } from 'react';
import './App.css';

function App() {
  const [extractedData, setExtractedData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileUpload = async (file) => {
    setLoading(true);
    setError(null);
    setExtractedData(null);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await fetch('http://localhost:5000/extract', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        throw new Error('Failed to extract data.');
      }
      const data = await response.json();
      setExtractedData(data.extracted_data || data.extractedData || data);
    } catch (err) {
      setError(err.message || 'An error occurred.');
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  return (
    <div className="app-container">
      <h1>OlmOCR Data Extraction</h1>
      
      {/* File Upload Component */}
      <div className="file-upload-container">
        <div
          className={`drop-area${loading ? ' loading' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => fileInputRef.current?.click()}
        >
          {loading ? (
            <span className="loader">Processing...</span>
          ) : (
            <>
              <span className="upload-icon">📄</span>
              <span>Drag & drop a PDF or image, or click to select</span>
            </>
          )}
        </div>
        <input
          type="file"
          accept=".pdf,.png,.jpg,.jpeg"
          ref={fileInputRef}
          style={{ display: 'none' }}
          onChange={handleFileChange}
          disabled={loading}
        />
      </div>

      {/* Error Display */}
      {error && <div className="error-message">{error}</div>}

      {/* Results Display */}
      {extractedData && !loading && (
        <div className="results-container">
          <h2>Extracted Data</h2>
          {extractedData.entities && (
            <div className="entities-section">
              <h3>Entities</h3>
              <div className="entity-list">
                <div>
                  <strong>Names:</strong> {extractedData.entities.names?.join(', ') || 'None'}
                </div>
                <div>
                  <strong>Dates:</strong> {extractedData.entities.dates?.join(', ') || 'None'}
                </div>
                <div>
                  <strong>Addresses:</strong> {extractedData.entities.addresses?.join(', ') || 'None'}
                </div>
              </div>
            </div>
          )}
          {extractedData.tables && extractedData.tables.length > 0 && (
            <div className="tables-section">
              <h3>Tables</h3>
              {extractedData.tables.map((table, idx) => (
                <table className="extracted-table" key={idx}>
                  <thead>
                    <tr>
                      {table.headers.map((header, i) => (
                        <th key={i}>{header}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {table.rows.map((row, i) => (
                      <tr key={i}>
                        {row.map((cell, j) => (
                          <td key={j}>{cell}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              ))}
            </div>
          )}
          {extractedData.raw_text && (
            <div className="raw-text-section">
              <h3>Raw Text</h3>
              <pre className="raw-text-block">{extractedData.raw_text}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
