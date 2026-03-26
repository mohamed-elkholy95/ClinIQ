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
import { MedicationExtractor } from './pages/MedicationExtractor';
import { AllergyExtractor } from './pages/AllergyExtractor';
import { VitalSigns } from './pages/VitalSigns';
import { QualityAnalyzer } from './pages/QualityAnalyzer';
import { SDoHExtractor } from './pages/SDoHExtractor';
import { ComorbidityCalculator } from './pages/ComorbidityCalculator';
import { Deidentification } from './pages/Deidentification';
import { PipelineExplorer } from './pages/PipelineExplorer';
import { TemporalExtractor } from './pages/TemporalExtractor';
import { AssertionDetector } from './pages/AssertionDetector';
import { RelationExplorer } from './pages/RelationExplorer';
import { DocumentClassifier } from './pages/DocumentClassifier';
import { SearchExplorer } from './pages/SearchExplorer';
import { DriftMonitor } from './pages/DriftMonitor';
import { StreamingAnalysis } from './pages/StreamingAnalysis';
import { ConversationMemory } from './pages/ConversationMemory';
import { EvaluationDashboard } from './pages/EvaluationDashboard';

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
              <Route path="/medications" element={<MedicationExtractor />} />
              <Route path="/allergies" element={<AllergyExtractor />} />
              <Route path="/vitals" element={<VitalSigns />} />
              <Route path="/quality" element={<QualityAnalyzer />} />
              <Route path="/sdoh" element={<SDoHExtractor />} />
              <Route path="/comorbidity" element={<ComorbidityCalculator />} />
              <Route path="/deidentify" element={<Deidentification />} />
              <Route path="/temporal" element={<TemporalExtractor />} />
              <Route path="/assertions" element={<AssertionDetector />} />
              <Route path="/relations" element={<RelationExplorer />} />
              <Route path="/classify" element={<DocumentClassifier />} />
              <Route path="/pipeline" element={<PipelineExplorer />} />
              <Route path="/search" element={<SearchExplorer />} />
              <Route path="/drift" element={<DriftMonitor />} />
              <Route path="/stream" element={<StreamingAnalysis />} />
              <Route path="/conversation" element={<ConversationMemory />} />
              <Route path="/evaluate" element={<EvaluationDashboard />} />
              <Route path="/timeline" element={<Timeline />} />
              <Route path="/models" element={<ModelManagement />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </ErrorBoundary>
    </QueryClientProvider>
  );
}
