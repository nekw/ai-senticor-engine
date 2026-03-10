# Mobile Responsiveness - Testing Checklist

## Quick Testing Guide

### Browser DevTools Testing (Recommended for Quick Check)

**Chrome/Edge:**
1. Press `F12` or `Ctrl+Shift+I` to open DevTools
2. Press `Ctrl+Shift+M` to toggle device toolbar
3. Select device presets from dropdown:
   - iPhone SE (375 x 667)
   - iPhone 12 Pro (390 x 844)
   - Samsung Galaxy S20 (360 x 800)
   - iPad Air (820 x 1180)
4. Test both portrait and landscape orientations

**Firefox:**
1. Press `F12` to open DevTools
2. Click the "Responsive Design Mode" icon (or `Ctrl+Shift+M`)
3. Select device presets or set custom dimensions

### Test Scenarios

#### ✅ Layout Tests
- [ ] Columns stack vertically on mobile (< 768px)
- [ ] Sidebar auto-collapses on mobile
- [ ] Metrics display in single column on small screens
- [ ] Charts fill container width
- [ ] Tables are horizontally scrollable

#### ✅ Component Tests
- [ ] **Trade Advisory Box**: Reduced padding and font size on mobile
- [ ] **Buttons**: Full width on mobile, minimum 44px height
- [ ] **Tabs**: Horizontally scrollable with appropriate font sizes
- [ ] **Plotly Charts**: Responsive and zoomable
- [ ] **Data Tables**: Scrollable with readable font sizes

#### ✅ Navigation Tests
- [ ] Sidebar hamburger menu accessible
- [ ] Tab navigation works with touch/swipe
- [ ] Expanders open/close correctly
- [ ] All buttons and controls are touch-friendly (44x44px minimum)

#### ✅ Content Tests
- [ ] Home tab displays correctly
- [ ] Market Intelligence map is viewable
- [ ] Company Intelligence loads properly
- [ ] Logs tab is readable
- [ ] PDF download works on mobile

#### ✅ Breakpoint Tests
Test at these specific widths:
- [ ] 1920px (Desktop)
- [ ] 1024px (Laptop)
- [ ] 768px (Tablet - breakpoint)
- [ ] 480px (Mobile - small breakpoint)
- [ ] 375px (iPhone SE)
- [ ] 360px (Small Android)

### Real Device Testing

**Setup:**
```bash
# 1. Find your computer's local IP
# Windows:
ipconfig
# Look for "IPv4 Address" (e.g., 192.168.1.100)

# macOS/Linux:
ifconfig
# Look for "inet" address

# 2. Run Streamlit with network access
streamlit run src/app.py --server.address 0.0.0.0

# 3. On your mobile device, visit:
# http://<your-ip>:8501
# Example: http://192.168.1.100:8501
```

**Test on:**
- [ ] iOS (iPhone/iPad)
- [ ] Android (Phone/Tablet)
- [ ] Different screen sizes
- [ ] Both portrait and landscape

### Performance Tests
- [ ] Page loads within 3 seconds on 4G
- [ ] Charts render smoothly
- [ ] Scrolling is smooth
- [ ] No horizontal overflow
- [ ] Images/components load properly

### Accessibility Tests
- [ ] Text is readable (minimum 16px body text on mobile)
- [ ] Touch targets are adequate (44x44px)
- [ ] Pinch-to-zoom works
- [ ] No overlapping elements
- [ ] Contrast ratios are maintained

## Common Issues & Solutions

### Issue: Sidebar doesn't collapse
**Solution:** Check `initial_sidebar_state="auto"` in `st.set_page_config()`

### Issue: Columns not stacking
**Solution:** Verify CSS media queries are loaded via `inject_mobile_styles()`

### Issue: Charts too small
**Solution:** Ensure `use_container_width=True` on all `st.plotly_chart()` calls

### Issue: Text too small to read
**Solution:** CSS should auto-adjust, check mobile breakpoints in `mobile_styles.py`

### Issue: Buttons hard to tap
**Solution:** Verify `min-height: 44px` and `min-width: 44px` in CSS

## Browser Compatibility

### Tested & Supported:
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+
- ✅ Samsung Internet
- ✅ Chrome Mobile (Android)
- ✅ Safari Mobile (iOS)

### Known Issues:
- Internet Explorer: Not supported
- Very old Android browsers (< 4.4): Limited support

## Debug Tips

1. **Check CSS Loading:**
   - Open DevTools → Elements → Search for "mobile-responsive"
   - Verify CSS rules are present

2. **Test Specific Breakpoint:**
   ```javascript
   // In browser console
   window.innerWidth  // Check current width
   ```

3. **Inspect Element:**
   - Right-click element → Inspect
   - Check computed styles
   - Verify media query is active

4. **Network Throttling:**
   - DevTools → Network → Throttling
   - Test on "Fast 3G" or "Slow 3G"

## Screenshots Location

Save test screenshots to: `docs/images/mobile_tests/`

Naming convention:
- `mobile_home_iphone12.png`
- `mobile_market_map_android.png`
- `tablet_company_intel_ipad.png`

## Reporting Issues

When reporting mobile issues, include:
1. Device type and OS version
2. Browser and version
3. Screen width (in px)
4. Screenshot
5. Steps to reproduce
6. Expected vs actual behavior

---

**Last Updated:** 2026-02-06
**Tested By:** [Your Name]
**Devices Tested:** [List devices]
