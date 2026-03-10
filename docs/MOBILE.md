# Mobile Responsive Features

The Senticor Engine UI is now fully optimized for mobile devices! 📱

## Key Mobile Enhancements

### 1. **Responsive Layout**
- Auto-stacking columns on mobile screens (< 768px)
- Optimized sidebar that auto-collapses on mobile
- Touch-friendly button sizes (minimum 44x44px)
- Responsive font sizes that scale with screen size

### 2. **Device Breakpoints**
- **Desktop**: > 768px - Full multi-column layouts
- **Tablet**: 481-768px - Adjusted layouts and font sizes
- **Mobile**: < 480px - Single-column stacked layouts

### 3. **Component Optimizations**

#### Charts & Visualizations
- Plotly charts scale to container width
- Minimum heights adjusted for mobile viewing
- Touch-enabled zoom and pan

#### Tables & Data
- Horizontal scrolling for data tables
- Reduced font sizes for better fit
- Container-width responsiveness

#### Navigation
- Collapsible sidebar for more screen real estate
- Scrollable tab navigation on small screens
- Compact tab labels on mobile

#### Trade Advisory Boxes
- Reduced padding on mobile (20px → 15px → 10px)
- Scaled font sizes (1.1rem → 0.95rem → 0.85rem)
- Maintained border and color coding

### 4. **Touch Interactions**
- Minimum touch target size compliance (44x44px)
- Smooth scrolling with `-webkit-overflow-scrolling: touch`
- Swipeable tabs and expandable sections

### 5. **Performance**
- CSS-only responsive design (no JavaScript required)
- Lazy loading maintained across all devices
- Optimized render performance

## Testing on Mobile

### Using Browser Dev Tools
1. Open Chrome/Edge DevTools (F12)
2. Click the device toggle (Ctrl+Shift+M)
3. Select a mobile device preset or set custom dimensions
4. Test different screen sizes:
   - iPhone SE (375px)
   - iPhone 12/13 (390px)
   - Samsung Galaxy (360px)
   - iPad (768px)

### Real Device Testing
Access the app on your mobile device:
```bash
# Run Streamlit with network access
streamlit run src/app.py --server.address 0.0.0.0
```

Then visit from your mobile device:
```
http://<your-computer-ip>:8501
```

## Mobile-Specific Features

### Auto-Collapse Sidebar
The sidebar automatically collapses on mobile, giving more space for content. Users can tap the hamburger menu to access controls.

### Stacked Metrics
Metric cards (Tickers, Sentiment, Volatility, etc.) automatically stack vertically on mobile for better readability.

### Responsive Charts
All Plotly charts use `use_container_width=True` for automatic scaling.

### Compact Controls
Buttons, inputs, and dropdowns are sized appropriately for mobile screens while maintaining touch-friendly dimensions.

## Best Practices for Mobile Users

1. **Portrait Mode**: Best experience in portrait orientation
2. **Zoom**: Pinch-to-zoom is enabled for detailed chart viewing
3. **Sidebar**: Use the hamburger menu (☰) to access controls
4. **Tabs**: Swipe horizontally to navigate between tabs
5. **Expandables**: Tap expanders for additional details

## Future Enhancements

Potential mobile improvements for future versions:
- [ ] Progressive Web App (PWA) support for install-to-home-screen
- [ ] Offline mode for cached data
- [ ] Dark mode optimization
- [ ] Voice input for ticker symbols
- [ ] Mobile-specific shortcuts/gestures
- [ ] Optimized PDF report generation for mobile viewing

## Technical Implementation

The mobile responsiveness is implemented using:
- **CSS Media Queries**: Breakpoint-based responsive design
- **Streamlit Native**: `use_container_width=True` for components
- **Custom CSS**: Injected via `mobile_styles.py`
- **Page Config**: Auto-collapsing sidebar via `initial_sidebar_state`

See [mobile_styles.py](../src/ui/mobile_styles.py) for the complete CSS implementation.
