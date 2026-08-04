import { apiGet } from './client';
import type {
  EntityProfile,
  EntityRelationsResponse,
  EntitySearchResponse,
} from '../types';

export function getEntity(id: string): Promise<EntityProfile> {
  return apiGet<EntityProfile>(`/api/entities/${id}`);
}

export function getEntityRelations(
  id: string,
): Promise<EntityRelationsResponse> {
  return apiGet<EntityRelationsResponse>(`/api/entities/${id}/relations`);
}

export function searchEntities(
  q: string,
  type?: string,
): Promise<EntitySearchResponse> {
  const params = new URLSearchParams({ q });
  if (type) params.set('type', type);
  return apiGet<EntitySearchResponse>(`/api/entities/search?${params}`);
}
