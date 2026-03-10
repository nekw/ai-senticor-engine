# 📱 Mobile Support - Quick Start Guide

## What Changed?

Your Senticor Engine now has **full mobile support**! The UI automatically adapts to any screen size - from desktop monitors to smartphones.

## Key Features

### 🎯 Automatic Responsive Layouts
- **Desktop (>768px)**: Multi-column layouts, expanded sidebar
- **Tablet (481-768px)**: Adjusted layouts, collapsible sidebar
- **Mobile (<480px)**: Single-column stacked, compact design

### 📊 Optimized Components
✅ Charts scale to fit screen width
✅ Tables scroll horizontally on mobile
✅ Buttons are full-width and touch-friendly (44x44px)
✅ Sidebar auto-collapses to hamburger menu
✅ Tabs scroll horizontally on small screens
✅ Metrics stack vertically on mobile

## How to Use

### On Desktop
Just run normally - no changes needed!
```bash
streamlit run src/app.py
```

### On Mobile Device
1. **Find your computer's IP address:**
   - Windows: Run `ipconfig` → look for "IPv4 Address"
   - Mac/Linux: Run `ifconfig` → look for "inet" address

2. **Run with network access:**
   ```bash
   streamlit run src/app.py --server.address 0.0.0.0
   ```

3. **Access from your mobile device:**
   - Open browser on your phone
   - Visit: `http://<your-ip>:8501`
   - Example: `http://192.168.1.100:8501`

### Quick Browser Test
1. Run the app normally
2. Open Chrome DevTools (F12)
3. Press Ctrl+Shift+M to toggle device mode
4. Select "iPhone 12 Pro" from dropdown
5. Navigate through the app!

## What You'll See on Mobile

### Home Tab
- Stats stack vertically instead of side-by-side
- Full-width buttons
- Readable text with optimized font sizes

### Market Intelligence
- Quadrant chart fills screen width
- Market summary table scrolls horizontally
- Controls stack for easy access

### Company Intelligence
- Ticker selector is full-width
- Metrics display vertically
- Charts scale to screen width
- News feed is easy to read

### Sidebar
- Auto-collapses to hamburger menu (☰)
- Tap menu to open/close
- All controls remain accessible
- Full-width buttons

## Files Added

### Core Implementation
- **`src/ui/mobile_styles.py`** - Complete responsive CSS stylesheet

### Documentation
- **`docs/MOBILE.md`** - Full mobile features guide
- **`docs/MOBILE_TESTING.md`** - Testing checklist
- **`docs/MOBILE_IMPLEMENTATION.md`** - Technical implementation details

### Modified Files
- `src/ui/config_loader.py` - Injects mobile CSS, auto-collapse sidebar
- `src/ui/components.py` - Mobile-friendly trade advisory
- `src/ui/tabs/deep_dive.py` - Responsive charts
- `src/ui/tabs/market_map.py` - Responsive layouts
- `README.md` - Added mobile support section

## Best Practices

### For Users
- ✅ Use portrait orientation for best experience
- ✅ Tap the hamburger menu (☰) to access sidebar
- ✅ Pinch-to-zoom works on charts
- ✅ Swipe horizontally to scroll tabs

### For Developers
- ✅ All `st.plotly_chart()` use `use_container_width=True`
- ✅ CSS handles responsiveness automatically
- ✅ Test at 375px, 768px, and 1920px widths
- ✅ Ensure 44x44px minimum touch targets

## Troubleshooting

### "Can't access from mobile"
- Ensure firewall allows port 8501
- Use `--server.address 0.0.0.0`
- Check you're on the same WiFi network

### "Sidebar doesn't collapse"
- This is automatic - works out of the box
- Tap the ☰ icon to toggle sidebar

### "Charts are too small"
- This shouldn't happen - all charts use `use_container_width=True`
- Try refreshing the page
- Check browser console for errors

### "Text is hard to read"
- Fonts auto-scale based on screen size
- Try portrait orientation
- Pinch-to-zoom if needed

## Testing

### Quick Test (2 minutes)
```bash
# 1. Run the app
streamlit run src/app.py

# 2. Open Chrome DevTools (F12)
# 3. Toggle device mode (Ctrl+Shift+M)
# 4. Select "iPhone 12 Pro"
# 5. Check all tabs work properly
```

### Full Test (15 minutes)
See complete checklist in `docs/MOBILE_TESTING.md`

## Browser Support

✅ **Fully Supported:**
- Chrome 90+ (Desktop & Mobile)
- Firefox 88+
- Safari 14+ (Desktop & iOS)
- Edge 90+
- Samsung Internet
- All modern mobile browsers

❌ **Not Supported:**
- Internet Explorer
- Very old Android browsers (<4.4)

## Performance

- ⚡ CSS-only implementation (no JavaScript overhead)
- ⚡ Fast page loads on 4G/5G
- ⚡ Smooth scrolling and interactions
- ⚡ Optimized for mobile networks

## Future Enhancements

Potential future mobile improvements:
- [ ] Progressive Web App (PWA) - install to home screen
- [ ] Offline mode
- [ ] Dark mode optimization
- [ ] Swipe gestures for navigation
- [ ] Voice input for tickers

## Need Help?

📖 **Full Documentation:**
- Features: [docs/MOBILE.md](MOBILE.md)
- Testing: [docs/MOBILE_TESTING.md](MOBILE_TESTING.md)
- Technical: [docs/MOBILE_IMPLEMENTATION.md](MOBILE_IMPLEMENTATION.md)

🐛 **Report Issues:**
Include:
- Device type and OS
- Browser and version
- Screen width
- Screenshot
- Steps to reproduce

---

**✨ Enjoy your mobile-optimized Senticor Engine!**

The UI now works seamlessly on any device - from desktop to smartphone. No additional configuration needed!
