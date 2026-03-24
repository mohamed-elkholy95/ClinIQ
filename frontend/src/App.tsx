import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { DocumentUpload } from './pages/DocumentUpload';
import { EntityViewer } from './pages/EntityViewer';
import { ICDResults } from './pages/ICDResults';
import { ClinicalSummary } from './pages/ClinicalSummary';
import { RiskAssessment } from './pages/RiskAssessment';
import { Timeline } from './pages/Timeline';
import { ModelManagement } from './pages/ModelManagement';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<DocumentUpload />} />
            <Route path="/entities" element={<EntityViewer />} />
            <Route path="/icd-codes" element={<ICDResults />} />
            <Route path="/summary" element={<ClinicalSummary />} />
            <Route path="/risk" element={<RiskAssessment />} />
            <Route path="/timeline" element={<Timeline />} />
            <Route path="/models" element={<ModelManagement />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
