import * as Sentry from "@sentry/react";

Sentry.init({
  dsn: import.meta.env.VITE_SENTRY_DSN || "",
  tracesSampleRate: 0.2,
  environment: import.meta.env.MODE,
  release: import.meta.env.VITE_APP_VERSION,
});

import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'

createRoot(document.getElementById("root")!).render(
  <Sentry.ErrorBoundary fallback={<p>An error has occurred. Please refresh.</p>}>
    <App />
  </Sentry.ErrorBoundary>
);
