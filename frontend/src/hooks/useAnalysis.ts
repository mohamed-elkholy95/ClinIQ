import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { analyzeText, extractEntities, predictICD, summarizeText, assessRisk } from '../services/api';
import type { AnalyzeRequest, NERRequest, ICDRequest, SummarizeRequest, RiskRequest } from '../types/api';

export function useAnalyze() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: AnalyzeRequest) => analyzeText(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      queryClient.invalidateQueries({ queryKey: ['dashboard'] });
    },
  });
}

export function useExtractEntities() {
  return useMutation({
    mutationFn: (data: NERRequest) => extractEntities(data),
  });
}

export function usePredictICD() {
  return useMutation({
    mutationFn: (data: ICDRequest) => predictICD(data),
  });
}

export function useSummarize() {
  return useMutation({
    mutationFn: (data: SummarizeRequest) => summarizeText(data),
  });
}

export function useAssessRisk() {
  return useMutation({
    mutationFn: (data: RiskRequest) => assessRisk(data),
  });
}

export function useAnalysisResult(id: string | undefined) {
  return useQuery({
    queryKey: ['analysis', id],
    queryFn: async () => {
      if (!id) throw new Error('No analysis ID');
      const { getDocument } = await import('../services/api');
      return getDocument(id);
    },
    enabled: !!id,
  });
}
