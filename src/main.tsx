import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import './index.css';
import { SettingsProvider } from './context/SettingsContext';
import { TopBarProvider } from './context/TopBarContext';

const root = document.getElementById('root');

if (root) {
  ReactDOM.createRoot(root).render(
    <React.StrictMode>
      <BrowserRouter>
        <SettingsProvider>
          <TopBarProvider>
            <App />
          </TopBarProvider>
        </SettingsProvider>
      </BrowserRouter>
    </React.StrictMode>
  );
}
