import React, { useState } from 'react';

function App() {
  return (
    <div className="app-container">
      <h1>OlmOCR Data Extraction</h1>
      <div style={{ textAlign: 'center', padding: '20px' }}>
        <p>Simple test app to verify React is working</p>
        <button 
          style={{ 
            padding: '10px 20px', 
            backgroundColor: '#2563eb', 
            color: 'white', 
            border: 'none', 
            borderRadius: '5px',
            cursor: 'pointer'
          }}
          onClick={() => alert('Button clicked!')}
        >
          Test Button
        </button>
      </div>
    </div>
  );
}

export default App;
