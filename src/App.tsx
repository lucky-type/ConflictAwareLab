import { Navigate, Route, Routes } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Environments from './pages/Environments';
import Algorithms from './pages/Algorithms';
import RewardFunctions from './pages/RewardFunctions';
import Models from './pages/Models';
import Experiments from './pages/Experiments';
import Layout from './components/layout/Layout';

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/environments" element={<Environments />} />
        <Route path="/algorithms" element={<Algorithms />} />
        <Route path="/reward-functions" element={<RewardFunctions />} />
        <Route path="/models" element={<Models />} />
        <Route path="/experiments" element={<Experiments />} />
      </Routes>
    </Layout>
  );
}

export default App;
