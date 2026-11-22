export interface BaseUrlResolution {
  current: string;
  appUrl: string | null;
  sharedUrl: string | null;
  recommendedUrl: string;
  isCurrentDevPreview: boolean;
  isCurrentSharedPreview: boolean;
  isRecommendedDevPreview: boolean;
  isRecommendedCookieGated: boolean;
}

export function isAiStudioDevPreviewUrl(urlStr: string | null | undefined): boolean {
  if (!urlStr) return false;
  try {
    const parsed = new URL(urlStr);
    return parsed.hostname.startsWith('ais-dev-');
  } catch {
    return /^https?:\/\/ais-dev-/i.test(urlStr);
  }
}

export function isAiStudioSharedPreviewUrl(urlStr: string | null | undefined): boolean {
  if (!urlStr) return false;
  try {
    const parsed = new URL(urlStr);
    return parsed.hostname.startsWith('ais-pre-');
  } catch {
    return /^https?:\/\/ais-pre-/i.test(urlStr);
  }
}

export function isCookieGatedPreviewUrl(urlStr: string | null | undefined): boolean {
  return isAiStudioDevPreviewUrl(urlStr) || isAiStudioSharedPreviewUrl(urlStr);
}

export function deriveSharedUrlFromDevUrl(urlStr: string | null | undefined): string | null {
  if (!urlStr) return null;
  try {
    const parsed = new URL(urlStr);
    if (parsed.hostname.startsWith('ais-dev-')) {
      parsed.hostname = `ais-pre-${parsed.hostname.slice('ais-dev-'.length)}`;
      return parsed.origin;
    }
    return null;
  } catch {
    return null;
  }
}

export function resolveBaseUrls(params: {
  requestOrigin: string;
  rawAppUrl?: string | null;
}): BaseUrlResolution {
  const current = params.requestOrigin.replace(/\/+$/, '');
  const appUrl =
    params.rawAppUrl &&
    params.rawAppUrl !== 'MY_APP_URL' &&
    /^https?:\/\//i.test(params.rawAppUrl)
      ? params.rawAppUrl.replace(/\/+$/, '')
      : null;

  const isCurrentDevPreview = isAiStudioDevPreviewUrl(current);
  const isCurrentSharedPreview = isAiStudioSharedPreviewUrl(current);
  const sharedUrl =
    deriveSharedUrlFromDevUrl(current) ||
    deriveSharedUrlFromDevUrl(appUrl) ||
    (!isCurrentDevPreview && /^https:\/\//i.test(current) ? current : null);

  let recommendedUrl = current;
  if (!isCurrentDevPreview) {
    recommendedUrl = current;
  } else if (sharedUrl) {
    recommendedUrl = sharedUrl;
  } else if (appUrl) {
    recommendedUrl = appUrl;
  }

  return {
    current,
    appUrl,
    sharedUrl,
    recommendedUrl,
    isCurrentDevPreview,
    isCurrentSharedPreview,
    isRecommendedDevPreview: isAiStudioDevPreviewUrl(recommendedUrl),
    isRecommendedCookieGated: isCookieGatedPreviewUrl(recommendedUrl),
  };
}
