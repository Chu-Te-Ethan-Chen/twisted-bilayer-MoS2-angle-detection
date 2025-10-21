#!/usr/bin/env python3
"""
Diagnostic script to analyze new MoS2 images and recommend optimal parameters
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def analyze_image_characteristics(image_path):
    """Analyze image characteristics and intensity distribution"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {Path(image_path).name}")
    print(f"{'='*80}")

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Failed to load image: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Image statistics
    print(f"\n📊 Image Statistics:")
    print(f"   Size: {img.shape[1]} x {img.shape[0]} pixels")
    print(f"   Channels: {img.shape[2] if len(img.shape) == 3 else 1}")
    print(f"   Data type: {img.dtype}")
    print(f"   File size: {Path(image_path).stat().st_size / 1024 / 1024:.2f} MB")

    # Intensity statistics
    print(f"\n💡 Intensity Statistics (Grayscale):")
    print(f"   Mean: {gray.mean():.2f}")
    print(f"   Median: {np.median(gray):.2f}")
    print(f"   Std Dev: {gray.std():.2f}")
    print(f"   Min: {gray.min()}")
    print(f"   Max: {gray.max()}")
    print(f"   Range: {gray.max() - gray.min()}")

    # Percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    print(f"\n📈 Intensity Percentiles:")
    for p in percentiles:
        print(f"   {p}th: {np.percentile(gray, p):.2f}")

    # Histogram analysis
    hist, bins = np.histogram(gray, bins=256, range=(0, 256))

    # Find potential thresholds
    # Look for bimodal distribution (background vs flakes)
    print(f"\n🎯 Suggested Detection Thresholds:")

    # Method 1: Mean - std
    threshold_1 = gray.mean() - gray.std()
    print(f"   Mean - 1σ: {threshold_1:.1f}")

    # Method 2: Otsu's method
    _, otsu_threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"   Otsu's method: {otsu_threshold:.1f}")

    # Method 3: 25th percentile
    threshold_25 = np.percentile(gray, 25)
    print(f"   25th percentile: {threshold_25:.1f}")

    # Method 4: Triangle method
    ret_triangle, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    print(f"   Triangle method: {ret_triangle:.1f}")

    # Color channel analysis
    if len(img.shape) == 3:
        print(f"\n🎨 RGB Channel Statistics:")
        for i, channel_name in enumerate(['Red', 'Green', 'Blue']):
            channel = img_rgb[:, :, i]
            print(f"   {channel_name}: Mean={channel.mean():.2f}, "
                  f"Std={channel.std():.2f}, Range=[{channel.min()}-{channel.max()}]")

    return {
        'path': str(image_path),
        'name': Path(image_path).name,
        'shape': img.shape,
        'gray_stats': {
            'mean': float(gray.mean()),
            'median': float(np.median(gray)),
            'std': float(gray.std()),
            'min': int(gray.min()),
            'max': int(gray.max())
        },
        'suggested_thresholds': {
            'mean_minus_std': float(threshold_1),
            'otsu': float(otsu_threshold),
            'percentile_25': float(threshold_25),
            'triangle': float(ret_triangle)
        },
        'img_rgb': img_rgb,
        'gray': gray
    }

def visualize_threshold_effects(analysis_results, output_dir):
    """Visualize how different thresholds affect detection"""

    for result in analysis_results:
        if result is None:
            continue

        img_rgb = result['img_rgb']
        gray = result['gray']
        name = result['name']

        # Test multiple thresholds
        test_thresholds = [
            ('Current (140)', 140),
            ('Otsu', result['suggested_thresholds']['otsu']),
            ('Mean-σ', result['suggested_thresholds']['mean_minus_std']),
            ('25th %ile', result['suggested_thresholds']['percentile_25']),
            ('Triangle', result['suggested_thresholds']['triangle']),
            ('Mean (adaptive)', gray.mean())
        ]

        fig, axes = plt.subplots(3, 3, figsize=(18, 18))

        # Original image
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # Grayscale
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('Grayscale', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # Histogram
        axes[0, 2].hist(gray.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
        axes[0, 2].axvline(140, color='red', linestyle='--', label='Current (140)')
        for label, thresh in test_thresholds[1:]:
            axes[0, 2].axvline(thresh, linestyle='--', alpha=0.5, label=f'{label}: {thresh:.1f}')
        axes[0, 2].set_title('Intensity Histogram', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Intensity')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend(fontsize=8)

        # Test different thresholds
        for idx, (label, threshold) in enumerate(test_thresholds):
            row = (idx + 3) // 3
            col = (idx + 3) % 3

            binary = (gray < threshold).astype(np.uint8) * 255

            # Apply morphological operations (same as in pipeline)
            kernel_open = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

            kernel_close = np.ones((4, 4), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter by area
            valid_contours = [c for c in contours if 200 <= cv2.contourArea(c) <= 15000]

            # Draw results
            result_img = img_rgb.copy()
            cv2.drawContours(result_img, valid_contours, -1, (0, 255, 0), 2)

            axes[row, col].imshow(result_img)
            axes[row, col].set_title(f'{label}: {threshold:.1f}\n{len(valid_contours)} flakes detected',
                                    fontsize=10, fontweight='bold')
            axes[row, col].axis('off')

        plt.tight_layout()

        # Save figure
        output_path = output_dir / f'{Path(name).stem}_threshold_diagnosis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved diagnostic visualization: {output_path}")
        plt.close()

def recommend_parameters(analysis_results):
    """Recommend optimal parameters based on analysis"""
    print(f"\n{'='*80}")
    print(f"🎯 PARAMETER RECOMMENDATIONS")
    print(f"{'='*80}")

    if not analysis_results or all(r is None for r in analysis_results):
        print("❌ No valid analysis results")
        return None

    # Filter out None results
    valid_results = [r for r in analysis_results if r is not None]

    # Aggregate statistics
    mean_intensities = [r['gray_stats']['mean'] for r in valid_results]
    otsu_thresholds = [r['suggested_thresholds']['otsu'] for r in valid_results]

    avg_mean = np.mean(mean_intensities)
    avg_otsu = np.mean(otsu_thresholds)

    print(f"\n📊 Aggregated Statistics Across {len(valid_results)} Images:")
    print(f"   Average mean intensity: {avg_mean:.1f}")
    print(f"   Average Otsu threshold: {avg_otsu:.1f}")
    print(f"   Current threshold: 140")
    print(f"   Difference: {140 - avg_mean:.1f} from mean")

    # Recommendations
    print(f"\n💡 Recommended Parameter Updates:")

    recommended_threshold = int(avg_otsu)
    print(f"\n1. intensity_threshold:")
    print(f"   Current: 140")
    print(f"   Recommended: {recommended_threshold}")
    print(f"   Reasoning: Based on Otsu's method average across all images")

    print(f"\n2. Additional adjustments:")
    if avg_mean > 150:
        print(f"   ⚠️  Images are brighter than reference (mean={avg_mean:.1f} vs ~120)")
        print(f"   → Consider using adaptive thresholding")
        print(f"   → May need to increase min_flake_area for noise reduction")
    elif avg_mean < 100:
        print(f"   ⚠️  Images are darker than reference (mean={avg_mean:.1f} vs ~120)")
        print(f"   → Lower threshold needed")
        print(f"   → May need more aggressive morphological filtering")
    else:
        print(f"   ✅ Images have similar brightness to reference")

    # Generate code snippet
    print(f"\n📝 Code Update Snippet:")
    print(f"```python")
    print(f"# Update in MoS2_UltraSensitive.ipynb, __init__ method:")
    print(f"self.intensity_threshold = {recommended_threshold}  # Updated from 140")
    print(f"```")

    return {
        'recommended_threshold': recommended_threshold,
        'avg_mean_intensity': avg_mean,
        'avg_otsu_threshold': avg_otsu
    }

def main():
    """Main diagnostic function"""
    print(f"\n{'='*80}")
    print(f"🔬 MoS2 Image Diagnostic Tool")
    print(f"{'='*80}")

    # Find new images
    data_dir = Path(__file__).parent / 'data' / 'raw_data'
    new_images = sorted(data_dir.glob('Capteur_E2V_*.tif'))

    print(f"\nFound {len(new_images)} new images to analyze:")
    for img in new_images:
        print(f"   - {img.name}")

    if not new_images:
        print("❌ No Capteur_E2V_*.tif images found in data/raw_data/")
        return

    # Analyze each image
    analysis_results = []
    for img_path in new_images:
        try:
            result = analyze_image_characteristics(str(img_path))
            analysis_results.append(result)
        except Exception as e:
            print(f"❌ Error analyzing {img_path.name}: {str(e)}")
            analysis_results.append(None)

    # Create output directory
    output_dir = Path(__file__).parent / 'results' / 'diagnostics'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize threshold effects
    print(f"\n{'='*80}")
    print(f"📊 Generating Diagnostic Visualizations")
    print(f"{'='*80}")
    visualize_threshold_effects(analysis_results, output_dir)

    # Generate recommendations
    recommendations = recommend_parameters(analysis_results)

    # Save recommendations to JSON
    if recommendations:
        recommendations_path = output_dir / 'parameter_recommendations.json'

        # Add individual image results
        recommendations['individual_results'] = []
        for r in analysis_results:
            if r is not None:
                recommendations['individual_results'].append({
                    'name': r['name'],
                    'gray_stats': r['gray_stats'],
                    'suggested_thresholds': r['suggested_thresholds']
                })

        with open(recommendations_path, 'w') as f:
            json.dump(recommendations, f, indent=2)

        print(f"\n💾 Saved recommendations to: {recommendations_path}")

    print(f"\n{'='*80}")
    print(f"✅ Diagnostic Analysis Complete!")
    print(f"{'='*80}")
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Review diagnostic images in {output_dir}")
    print(f"2. Update MoS2_UltraSensitive.ipynb with recommended parameters")
    print(f"3. Re-run the pipeline on your new images")

if __name__ == '__main__':
    main()
