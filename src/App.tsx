import { Navigate, Route, Routes } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Environments from './pages/Environments';
import Agents from './pages/Agents';
import ResidualConnectors from './pages/ResidualConnectors';
import RewardFunctions from './pages/RewardFunctions';
import Models from './pages/Models';
import Experiments from './pages/Experiments';
// import Curriculum from './pages/Curriculum';
import Layout from './components/layout/Layout';
import { DataCacheProvider } from './context/DataCacheContext';

function App() {
  return (
    <DataCacheProvider>
      <Layout>
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/environments" element={<Environments />} />
          <Route path="/agents" element={<Agents />} />
          <Route path="/residual-connectors" element={<ResidualConnectors />} />
          <Route path="/reward-functions" element={<RewardFunctions />} />
          <Route path="/models" element={<Models />} />
          <Route path="/experiments" element={<Experiments />} />
          {/* <Route path="/curriculum" element={<Curriculum />} /> */}
        </Routes>
      </Layout>
    </DataCacheProvider>
  );
}

export default App;
