// Move all imports to the top
import { UserProfile, UserProfileUpdate } from '@/types/userTypes';
import { PersonalizedWardrobeSuggestions } from '@/types/recommendationTypes';
import { Feedback, FeedbackCreate } from '@/types/outfitTypes';

// Base URL from environment or fallback
const BASE_URL = 'http://localhost:8000';

// Helper: get auth token from localStorage
const getAuthToken = () => localStorage.getItem('token');

// Generic API client
interface RequestOptions {
  method?: string;
  headers?: Record<string, string>;
  body?: any;
}

const apiClient = async (endpoint: string, options: RequestOptions = {}) => {
  const { method = 'GET', body } = options;
  const token = getAuthToken();

  const headers: Record<string, string> = { ...(options.headers || {}) };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  // JSON content-type when needed
  if (body && typeof body === 'object' && !(body instanceof FormData) && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }

  const config: RequestInit = {
    method,
    headers,
    body: headers['Content-Type'] === 'application/json' && body ? JSON.stringify(body) : body,
  };

  const url = `${BASE_URL}/api${endpoint}`;

  try {
    const res = await fetch(url, config);
    if (res.status === 204) return null;
    const data = await res.json().catch(() => null);
    if (!res.ok) {
      const msg = data?.detail || data?.message || data?.error || res.statusText;
      throw new Error(msg);
    }
    return data;
  } catch (err: any) {
    console.error('API Client Error:', err.message);
    throw err;
  }
};

export default apiClient;

// Types
export interface WardrobeItemData {
  name?: string;
  brand?: string;
  category?: string;
  size?: string;
  price?: number;
  material?: string;
  season?: string;
  image_url?: string | null;
  tags?: string[];
  color?: string;
  notes?: string;
  favorite?: boolean;
}

// CRUD for wardrobe items


export const addItem = async (
  itemData: Omit<WardrobeItemData, 'image_url'>,
  imageFile?: File
) => {
  console.log("[apiClient.addItem] Raw itemData to be stringified:", itemData);
  const itemDataString = JSON.stringify(itemData);
  console.log("[apiClient.addItem] Stringified itemData:", itemDataString);

  const formData = new FormData();
  formData.append("item", itemDataString); // ✅ FastAPI expects item as string

  if (imageFile) {
    formData.append("image", imageFile, imageFile.name);
    console.log("[apiClient.addItem] Appended image file:", imageFile.name, imageFile.type, imageFile.size);
  } else {
    console.log("[apiClient.addItem] No image file provided.");
  }

  // To inspect FormData, iterate over entries (won't show file content but will show keys/filenames)
  // console.log("[apiClient.addItem] FormData entries:");
  // for (let pair of formData.entries()) {
  //   console.log(pair[0]+ ', ' + (pair[1] instanceof File ? pair[1].name : pair[1]));
  // }

  try {
    // Use apiClient helper — DO NOT set Content-Type, FormData will handle it
    const response = await apiClient("/wardrobe/items/", {
      method: "POST",
      body: formData,
    });
    console.log("[apiClient.addItem] Success response:", response);
    return response;
  } catch (error: any) {
    console.error("[apiClient.addItem] Error caught in apiClient.addItem:", error);
    // Log more details if available, e.g., error.response for axios-like errors
    if (error.response) {
      console.error("[apiClient.addItem] Error response data:", error.response.data);
      console.error("[apiClient.addItem] Error response status:", error.response.status);
    }
    throw error; // Re-throw the error to be caught by the caller
  }
};



export const updateItem = (itemId: string, itemData: WardrobeItemData, imageFile?: File) => {
  const form = new FormData();
  form.append('item_update', new Blob([JSON.stringify(itemData)], { type: 'application/json' }));
  if (imageFile) form.append('image', imageFile);
  return apiClient(`/wardrobe/items/${itemId}`, { method: 'PUT', body: form });
};

// User profile
export const getUserProfile = (): Promise<UserProfile> => apiClient('/profile/me');
export const updateUserProfile = (profileData: UserProfileUpdate): Promise<UserProfile> =>
  apiClient('/profile/me', { method: 'PUT', body: profileData });

// Recommendations
export const getWardrobeSuggestions = (
  latitude?: number,
  longitude?: number
): Promise<PersonalizedWardrobeSuggestions> => {
  const params = new URLSearchParams();
  if (latitude != null) params.append('lat', latitude.toString());
  if (longitude != null) params.append('lon', longitude.toString());
  const query = params.toString();
  return apiClient(`/recommendations/wardrobe/${query ? `?${query}` : ''}`);
};

// Feedback
export const getFeedbackForOutfit = (id: string | number): Promise<Feedback[]> =>
  apiClient(`/community/outfits/${id}/feedback`);

export const addFeedbackToOutfit = (
  id: string | number,
  feedbackData: FeedbackCreate
): Promise<Feedback> => apiClient(`/community/outfits/${id}/feedback`, { method: 'POST', body: feedbackData });

export const deleteFeedback = (feedbackId: number) =>
  apiClient(`/community/feedback/${feedbackId}`, { method: 'DELETE' });

// ML Features API endpoints

export interface ClassificationResult {
  predicted_category: string;
  confidence_score: number;
  detailed_predictions: Record<string, number>;
  model_name: string;
}

export interface ColorAnalysisResult {
  dominant_color_rgb: number[];
  dominant_color_hex: string;
  dominant_color_name: string;
  color_properties: Record<string, string>;
}

export interface ColorPaletteResult {
  palette: Array<{
    rgb: number[];
    hex: string;
    name: string;
    properties: Record<string, string>;
  }>;
  dominant_color: any;
  harmony_analysis: Record<string, string>;
  color_count: number;
}

export interface UserPreferences {
  preferred_colors?: string[];
  avoided_colors?: string[];
  preferred_styles?: string[];
  preferred_brands?: string[];
  size_preferences?: Record<string, string>;
  budget_range?: [number, number];
}

export interface RecommendationRequest {
  user_preferences?: UserPreferences;
  occasion?: string;
  num_recommendations?: number;
  weather_conditions?: Record<string, any>;
  color_scheme?: string;
}

export interface OutfitRecommendation {
  id?: number;
  name?: string;
  items: Array<{
    id: number;
    name: string;
    category: string;
    brand?: string;
    dominant_color_name?: string;
    image_url?: string;
    similarity_score?: number;
  }>;
  score: number;
  color_score?: number;
  style_score?: number;
  occasion_score?: number;
  recommendation_reason?: string;
}

export interface SimilarItem {
  id: number;
  name: string;
  category: string;
  brand?: string;
  dominant_color_name?: string;
  image_url?: string;
  similarity_score: number;
  similarity_reasons?: string[];
}

// ML API functions

/**
 * Classify a clothing item image using MobileNetV2
 */
export const classifyClothingImage = async (imageFile: File): Promise<ClassificationResult> => {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  return apiClient('/ml/classify-clothing', {
    method: 'POST',
    body: formData
  });
};

/**
 * Detect dominant color from a clothing item image
 */
export const detectDominantColor = async (imageFile: File): Promise<ColorAnalysisResult> => {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  return apiClient('/ml/detect-dominant-color', {
    method: 'POST',
    body: formData
  });
};

/**
 * Extract color palette from a clothing item image
 */
export const extractColorPalette = async (
  imageFile: File, 
  colorCount: number = 5
): Promise<ColorPaletteResult> => {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  return apiClient(`/ml/extract-color-palette?color_count=${colorCount}`, {
    method: 'POST',
    body: formData
  });
};

/**
 * Get outfit recommendations based on user preferences and occasion
 */
export const getOutfitRecommendations = async (
  request: RecommendationRequest
): Promise<OutfitRecommendation[]> => {
  return apiClient('/ml/outfit-recommendations', {
    method: 'POST',
    body: request
  });
};

/**
 * Get items similar to a target item
 */
export const getSimilarItems = async (
  targetItemId: number,
  numSimilar: number = 5
): Promise<SimilarItem[]> => {
  return apiClient(`/ml/similar-items?target_item_id=${targetItemId}&num_similar=${numSimilar}`, {
    method: 'POST'
  });
};

/**
 * Train the recommendation model with user's wardrobe
 */
export const trainRecommendationModel = async (): Promise<{
  status: string;
  message: string;
}> => {
  return apiClient('/ml/train-recommendation-model', {
    method: 'POST'
  });
};

/**
 * Enhanced add item function with ML processing
 */
export const addItemWithML = async (
  itemData: WardrobeItemData, 
  imageFile?: File
): Promise<any> => {
  const form = new FormData();
  form.append('item', new Blob([JSON.stringify(itemData)], { type: 'application/json' }));
  if (imageFile) {
    form.append('image', imageFile);
  }
  
  // First add the item normally
  const result = await apiClient('/wardrobe/items/', { method: 'POST', body: form });
  
  // If image was provided, process ML features
  if (imageFile && result?.id) {
    try {
      // Run classification and color detection in parallel
      const [classificationResult, colorResult] = await Promise.allSettled([
        classifyClothingImage(imageFile),
        detectDominantColor(imageFile)
      ]);
      
      // Update the item with ML results
      const mlData: Partial<WardrobeItemData> = {};
      
      if (classificationResult.status === 'fulfilled') {
        // Update category if classification is confident enough
        if (classificationResult.value.confidence_score > 0.7) {
          mlData.category = classificationResult.value.predicted_category;
        }
      }
      
      if (colorResult.status === 'fulfilled') {
        // Add color information to tags or notes
        const colorName = colorResult.value.dominant_color_name;
        mlData.tags = [...(itemData.tags || []), colorName];
      }
      
      // Update the item if we have ML data
      if (Object.keys(mlData).length > 0) {
        await updateItem(result.id.toString(), mlData);
      }
      
    } catch (error) {
      console.warn('ML processing failed, but item was added successfully:', error);
    }
  }
  
  return result;
};

/**
 * Get color suggestions for coordinating with an item
 */
export const getColorSuggestions = async (dominantColorRgb: number[]): Promise<{
  complementary: Array<{ rgb: number[]; hex: string; name: string; }>;
  analogous: Array<{ rgb: number[]; hex: string; name: string; }>;
  neutrals: Array<{ rgb: number[]; hex: string; name: string; }>;
}> => {
  // This would be implemented as a separate endpoint or computed client-side
  // For now, return a placeholder
  return {
    complementary: [],
    analogous: [],
    neutrals: []
  };
};

/**
 * Get smart outfit suggestions for a specific occasion
 */
export const getSmartOutfitSuggestions = async (
  occasion: string,
  weatherConditions?: Record<string, any>,
  userPreferences?: UserPreferences
): Promise<OutfitRecommendation[]> => {
  const request: RecommendationRequest = {
    occasion,
    weather_conditions: weatherConditions,
    user_preferences: userPreferences,
    num_recommendations: 5
  };
  
  return getOutfitRecommendations(request);
};

/**
 * Analyze wardrobe and get insights
 */
export const getWardrobeInsights = async (): Promise<{
  color_distribution: Record<string, number>;
  category_distribution: Record<string, number>;
  style_analysis: Record<string, any>;
  recommendations: string[];
}> => {
  // This would be a new endpoint that analyzes the user's entire wardrobe
  // For now, return a placeholder
  return {
    color_distribution: {},
    category_distribution: {},
    style_analysis: {},
    recommendations: []
  };
};

