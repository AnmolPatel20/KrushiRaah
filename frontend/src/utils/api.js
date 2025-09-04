const apiBase = process.env.REACT_APP_API_URL
  ? (process.env.REACT_APP_API_URL.startsWith('http')
      ? process.env.REACT_APP_API_URL
      : `https://${process.env.REACT_APP_API_URL}`)
  : '';

export const apiUrl = (path) => {
  if (!apiBase) return path;
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${apiBase}${normalizedPath}`;
};

export default apiUrl;

