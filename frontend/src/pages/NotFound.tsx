/**
 * 404 Not Found page.
 *
 * Displayed when a user navigates to a route that doesn't match any
 * defined page.  Uses the same visual language as the ErrorBoundary
 * fallback (icon + heading + body + action button) so error states
 * feel consistent across the application.
 *
 * Accessibility: the main heading is an h1, the "Go Home" link uses
 * a visible focus ring, and the decorative icon is hidden from screen
 * readers via aria-hidden.
 */

import { Link } from 'react-router-dom';
import { Home, MapPinOff } from 'lucide-react';

export function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] px-6 text-center">
      {/* Decorative icon */}
      <div className="rounded-full bg-primary-100 dark:bg-primary-900/30 p-5 mb-6">
        <MapPinOff
          className="w-10 h-10 text-primary-500 dark:text-primary-400"
          aria-hidden="true"
        />
      </div>

      {/* Heading */}
      <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-2">
        Page not found
      </h1>

      {/* Subheading */}
      <p className="text-base text-gray-600 dark:text-gray-400 mb-8 max-w-md">
        The page you're looking for doesn't exist or has been moved.
        Check the URL or head back to the dashboard.
      </p>

      {/* Action */}
      <Link
        to="/"
        className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-medium text-white bg-primary-500 hover:bg-primary-600 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900"
      >
        <Home className="w-4 h-4" />
        Back to Dashboard
      </Link>
    </div>
  );
}
