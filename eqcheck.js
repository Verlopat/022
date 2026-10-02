// Render the equations of a docx (via pandoc MathML HTML) in Chromium and save a screenshot.
const { chromium } = require('playwright');
(async () => {
  const [html, png] = process.argv.slice(2);
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1000, height: 800 } });
  await p.goto('file://' + html);
  await p.screenshot({ path: png, fullPage: true });
  await b.close();
})();
