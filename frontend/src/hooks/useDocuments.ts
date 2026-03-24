import { useQuery } from '@tanstack/react-query';
import { getDocuments, getDocument } from '../services/api';

export function useDocuments(page = 1, perPage = 20) {
  return useQuery({
    queryKey: ['documents', page, perPage],
    queryFn: () => getDocuments(page, perPage),
    staleTime: 30_000,
  });
}

export function useDocument(id: string | undefined) {
  return useQuery({
    queryKey: ['document', id],
    queryFn: () => getDocument(id!),
    enabled: !!id,
  });
}
