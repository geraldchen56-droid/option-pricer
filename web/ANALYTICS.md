# Vercel Web Analytics Configuration

This document describes the Vercel Web Analytics implementation for the Root T project.

## Implementation Details

### Current Setup

The project uses the **script tag approach** for Vercel Web Analytics, which is the recommended method for static HTML sites deployed on Vercel.

### Code Location

The analytics code is located in `web/index.html` in the `<head>` section (lines 75-76):

```html
<script>window.va=window.va||function(){(window.vaq=window.vaq||[]).push(arguments)};</script>
<script defer src="/_vercel/insights/script.js"></script>
```

### How It Works

1. **Analytics Stub** (line 75): Creates the `window.va` function queue that buffers analytics calls before the main script loads
2. **Analytics Script** (line 76): Loads the actual Vercel Web Analytics script from Vercel's infrastructure

### Why This Approach?

For static HTML sites (sites without a build process or npm), Vercel recommends using script tags rather than the `@vercel/analytics` npm package. This approach:

- ✅ Requires no build step
- ✅ Requires no package manager or dependencies
- ✅ Automatically loads the latest analytics version from Vercel
- ✅ Works immediately when Analytics is enabled in the Vercel dashboard

### Enabling Analytics

To see analytics data:

1. Deploy this site to Vercel
2. Navigate to your project in the Vercel dashboard
3. Go to the "Analytics" tab
4. Click "Enable Web Analytics"
5. Analytics will start collecting data automatically

No code changes are needed - the script tags are already in place.

### Verification

After deployment, verify analytics is working by:

1. Visit your deployed site
2. Open browser DevTools → Network tab
3. Look for a request to `/_vercel/insights/view` or similar
4. Check the Vercel dashboard Analytics tab (data may take a few minutes to appear)

### Documentation References

- [Vercel Web Analytics Quickstart](https://vercel.com/docs/analytics/quickstart)
- [Vercel Web Analytics Overview](https://vercel.com/docs/analytics)

### Alternative Approaches Not Used

**NPM Package Method**: For projects with a build system (React, Next.js, etc.), you can use:
```bash
npm install @vercel/analytics
```

This is **not needed** for this project since it's a static HTML site. The script tag method is simpler and equally effective.

## Speed Insights

Note: This project currently has **Web Analytics** (visitor tracking) but not **Speed Insights** (performance metrics). These are separate products.

To add Speed Insights, you would add:
```html
<script>window.si=window.si||function(){(window.siq=window.siq||[]).push(arguments)};</script>
<script defer src="/_vercel/speed-insights/script.js"></script>
```

## Summary

✅ Vercel Web Analytics is **already configured and ready to use**
✅ No additional installation steps required
✅ Simply enable in the Vercel dashboard to start collecting data
