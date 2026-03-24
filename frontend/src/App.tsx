import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ErrorBoundary from './components/ErrorBoundary';
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

/**
 * Root application component.
 *
 * The ErrorBoundary wraps the entire route tree so any unhandled rendering
 * error shows a friendly fallback instead of a white screen.  Individual
 * pages can add their own nested boundaries for more granular recovery.
 */
export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ErrorBoundary>
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
      </ErrorBoundary>
    </QueryClientProvider>
  );
}
