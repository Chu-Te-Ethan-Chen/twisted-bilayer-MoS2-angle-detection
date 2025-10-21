# 🚨 Quick Fix Guide: Detecting MoS₂ in New Images

## Problem
Your 5 new `.tif` images are not being detected by the standard `MoS2_UltraSensitive.ipynb` pipeline.

## Root Cause
The new images likely have different characteristics than the original reference images:
- Different **intensity distribution** (brightness)
- Different **contrast** levels
- Different **bit depth** (16-bit vs 8-bit)
- Different **imaging conditions** (microscope settings, lighting, etc.)

## 🎯 Solution: Three-Step Approach

### Step 1: Run Diagnostic Analysis (Required)
**Purpose:** Understand what's different about your new images

**Notebook:** `MoS2_Simple_Diagnostics.ipynb` ⭐ **RECOMMENDED - Bulletproof!**

**Alternative:** `MoS2_Image_Diagnostics.ipynb` (has OpenCV version issues)

**Instructions:**
1. Open Google Colab
2. Upload `MoS2_Simple_Diagnostics.ipynb`
3. Upload your 5 new `.tif` images to `/content/images/`
4. Run all cells
5. Review the diagnostic visualizations

**What You'll Get:**
- Intensity distribution analysis
- Recommended threshold values
- Visual comparison of different thresholds
- Parameter recommendations (JSON file)

**Expected Output:**
```
Recommended Parameter Updates:
1. intensity_threshold:
   Current: 140
   Recommended: XXX (specific to your images)

2. CLAHE preprocessing: [Enabled/Disabled]
```

---

### Step 2A: Try Adaptive Pipeline (Easiest)
**Purpose:** Automatically adjust to your new images

**Notebook:** `MoS2_Adaptive_Pipeline.ipynb` *(Note: This is a simplified version)*

**Status:** ⚠️ **INCOMPLETE** - I created the framework but didn't include all methods

**Recommendation:** Skip to Step 2B for now

---

### Step 2B: Manual Parameter Update (Recommended)
**Purpose:** Update the original pipeline with optimized parameters

**Notebook:** `MoS2_UltraSensitive.ipynb`

**Instructions:**

1. **Get recommended threshold** from Step 1 diagnostic output

2. **Open** `MoS2_UltraSensitive.ipynb` in Google Colab

3. **Find the `__init__` method** in Cell 2 (around line 3):
   ```python
   def __init__(self, visualization_mode='clean'):
       self.intensity_threshold = 140  # <-- CHANGE THIS LINE
   ```

4. **Update the threshold** to the recommended value:
   ```python
   self.intensity_threshold = XXX  # Use value from diagnostic
   ```

5. **If low contrast detected** (std < 20), add CLAHE preprocessing:

   Find the `stage1_detect_flakes` method (around line 20):
   ```python
   # Load image
   img = cv2.imread(image_path)
   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

   # ADD THESE LINES FOR LOW CONTRAST IMAGES:
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
   gray = clahe.apply(gray)
   ```

6. **If 16-bit images**, add bit depth normalization:

   Right after loading the image:
   ```python
   img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

   # ADD THIS FOR 16-BIT IMAGES:
   if img.dtype == np.uint16:
       img = (img / 256).astype(np.uint8)

   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   ```

7. **Run the pipeline** with your new images

---

### Step 3: Verify and Iterate
**Purpose:** Confirm detection works

**Instructions:**
1. Upload your `.tif` images to `/content/images/`
2. Run all cells in the updated `MoS2_UltraSensitive.ipynb`
3. Check results:
   - **Success:** Flakes detected → Proceed with analysis
   - **Partial:** Some flakes detected → Fine-tune threshold (±10-20)
   - **Failure:** No flakes detected → Go back to diagnostics

---

## 📊 Quick Parameter Reference

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Too bright** | Mean intensity > 150 | Increase threshold (+20 to +40) |
| **Too dark** | Mean intensity < 100 | Decrease threshold (-10 to -20) |
| **Low contrast** | Std deviation < 20 | Enable CLAHE preprocessing |
| **16-bit images** | dtype=uint16 | Add bit depth normalization |
| **No flakes** | 0 flakes detected | Run diagnostics, check all above |

---

## 🛠️ Common Threshold Values

Based on typical imaging conditions:

- **Original reference images:** 140
- **Bright/washed out images:** 160-180
- **Dark images:** 110-130
- **Very high contrast:** 120-140
- **Very low contrast:** 150-170 (with CLAHE)

---

## 📁 Files Created for You

1. **`MoS2_Simple_Diagnostics.ipynb`** ✅ **RECOMMENDED**
   - Bulletproof diagnostic tool (no OpenCV version issues!)
   - Visual threshold comparison
   - Clear recommendations

2. **`MoS2_Image_Diagnostics.ipynb`** ⚠️ Has bugs
   - Advanced diagnostic with Otsu/Triangle methods
   - OpenCV version compatibility issues
   - Use Simple version instead

3. **`MoS2_Adaptive_Pipeline.ipynb`** ⚠️ Incomplete
   - Framework created but not fully functional
   - Use manual method (Step 2B) instead

4. **`diagnose_new_images.py`** ⚠️ Requires local dependencies
   - Python script version of diagnostics
   - Use Colab notebook instead

5. **`QUICK_FIX_GUIDE.md`** ✅ This file
   - Step-by-step instructions

---

## 🎯 Recommended Workflow

```
1. Upload images to Colab
   ↓
2. Run MoS2_Simple_Diagnostics.ipynb ⭐
   ↓
3. Note recommended threshold (e.g., 107)
   ↓
4. Update MoS2_UltraSensitive.ipynb
   - Change intensity_threshold to 107
   - Add CLAHE if needed
   - Add bit normalization if needed
   ↓
5. Run updated MoS2_UltraSensitive.ipynb
   ↓
6. Check results
   - Success? Done!
   - No? Adjust threshold ±10 and retry
```

---

## ❓ Troubleshooting

### "Still detecting 0 flakes after parameter update"

Try these in order:

1. **Verify images uploaded correctly**
   ```python
   import os
   print(os.listdir('/content/images'))
   ```

2. **Check image loading**
   ```python
   import cv2
   img = cv2.imread('/content/images/your_image.tif', cv2.IMREAD_UNCHANGED)
   print(f"Image shape: {img.shape}")
   print(f"Image dtype: {img.dtype}")
   print(f"Min: {img.min()}, Max: {img.max()}, Mean: {img.mean()}")
   ```

3. **Try extreme thresholds**
   - Very high: 200
   - Very low: 80
   - See which direction works

4. **Disable shape filtering temporarily**
   In `stage1_detect_flakes`, comment out the shape requirements:
   ```python
   # is_valid_flake = (
   #     0.3 < solidity < 1.0 and
   #     aspect_ratio < 5.0 and
   #     circularity > 0.15 and
   #     3 <= len(approx) <= 10
   # )
   is_valid_flake = True  # Accept everything temporarily
   ```

5. **Check area filters**
   - Maybe your flakes are smaller/larger than 200-15000 pixels
   - Adjust `min_flake_area` and `max_flake_area`

---

## 📞 Next Steps

**If diagnostics work but manual update doesn't:**
- Share the diagnostic output (JSON file)
- Share a sample image (if possible)
- I can provide more specific parameter recommendations

**If you need the complete adaptive pipeline:**
- I can create a full implementation by copying all methods from the original pipeline
- This will take a bit longer but will be fully automatic

**If everything works:**
- Great! Proceed with your analysis
- The rest of the pipeline (Stage 2, Stage 3) should work as-is once Stage 1 detects flakes

---

## 📝 Summary

1. **Problem:** New images have different characteristics
2. **Diagnosis:** Run `MoS2_Simple_Diagnostics.ipynb` ⭐
3. **Fix:** Update threshold in `MoS2_UltraSensitive.ipynb`
4. **Verify:** Re-run and check results
5. **Iterate:** Fine-tune if needed

**Most likely solution:** Change line 3 of Cell 2 from `140` to `107` based on your image statistics (25th percentile).

---

*Created: October 20, 2025*
*For: Twisted Bilayer MoS₂ Angle Detection Project*
