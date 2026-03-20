import { useState } from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { PreviewArea } from './components/PreviewArea';
import { ControlPanel } from './components/ControlPanel';
import { ThemeProvider } from './contexts/ThemeContext';

// Base URL of your FastAPI backend
const API_BASE = 'http://localhost:8000';

// All treatments grouped by category (must match backend config.py)
const TREATMENTS_BY_CATEGORY: Record<string, string[]> = {
  Face: ['temples_fillers', 'cheek_fillers', 'chin_fillers', 'jawline_contouring', 'forehead_lines', 'glabellar_lines', 'nasolabial_folds', 'marionette_folds'],
  Lips: ['plumper', 'cupids_bow', 'upper_lip_fillers', 'lower_lip_fillers', 'corner_lip_lift_fillers'],
  Nose: ['contouring', 'bridge_fillers', 'root_fillers', 'tip_lift_fillers', 'slimming_fillers'],
  Eyebrows: ['brow_lift'],
};

export default function App() {
  // ── UI state ──────────────────────────────────────────────────────
  const [selectedCategory, setSelectedCategory] = useState('Face');
  const [selectedTreatment, setSelectedTreatment] = useState('temples_fillers');
  const [intensity, setIntensity] = useState(60);
  const [xPosition, setXPosition] = useState(0);
  const [yPosition, setYPosition] = useState(0);

  // ── Image state ───────────────────────────────────────────────────
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);  // original photo URL
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);         // transformed photo URL
  const [imageId, setImageId] = useState<string | null>(null);               // ID returned by backend after upload

  // ── Loading / error state ─────────────────────────────────────────
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // ── Handlers ──────────────────────────────────────────────────────

  // When the user switches category, also reset the treatment to the first one in that category
  const handleSelectCategory = (category: string) => {
    setSelectedCategory(category);
    setSelectedTreatment(TREATMENTS_BY_CATEGORY[category][0]);
    setPreviewUrl(null); // clear preview when switching category
  };

  // Step 1: Upload the photo to the backend and get back an image_id
  const handleUploadImage = async (file: File) => {
    setIsLoading(true);
    setErrorMessage(null);
    setPreviewUrl(null); // clear any old preview

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE}/api/upload/image`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setImageId(data.image_id);
        setUploadedImage(`${API_BASE}${data.image_url}`); // show original photo
      } else {
        setErrorMessage('Upload failed: ' + data.message);
      }
    } catch {
      setErrorMessage('Could not connect to backend. Make sure it is running on port 8000.');
    } finally {
      setIsLoading(false);
    }
  };

  // Step 2a: Fast preview — downscaled image, quick result
  const handlePreview = async () => {
    if (!imageId) {
      setErrorMessage('Please upload an image first.');
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);

    try {
      const response = await fetch(`${API_BASE}/api/transform/apply`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_id: imageId,
          category: selectedCategory.toLowerCase(),   // backend expects lowercase: "face", "lips", etc.
          treatment: selectedTreatment,
          intensity: intensity / 100,                 // slider is 0-100, backend expects 0.0-1.0
          position_x: xPosition / 50,                 // slider is -50 to 50, backend expects -1.0 to 1.0
          position_y: yPosition / 50,
          preview: true,                              // fast low-res preview mode
        }),
      });

      const data = await response.json();

      if (data.success) {
        // Add a timestamp to bust the browser cache so the new image always loads
        setPreviewUrl(`${API_BASE}${data.preview_url}?t=${Date.now()}`);
      } else {
        setErrorMessage('Transform failed: ' + data.message);
      }
    } catch {
      setErrorMessage('Could not connect to backend. Make sure it is running on port 8000.');
    } finally {
      setIsLoading(false);
    }
  };

  // Step 2b: Apply Final — full resolution, slower but high quality
  const handleApplyFinal = async () => {
    if (!imageId) {
      setErrorMessage('Please upload an image first.');
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);

    try {
      const response = await fetch(`${API_BASE}/api/transform/apply`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_id: imageId,
          category: selectedCategory.toLowerCase(),
          treatment: selectedTreatment,
          intensity: intensity / 100,
          position_x: xPosition / 50,
          position_y: yPosition / 50,
          preview: false,                             // full resolution mode
        }),
      });

      const data = await response.json();

      if (data.success) {
        setPreviewUrl(`${API_BASE}${data.preview_url}?t=${Date.now()}`);
      } else {
        setErrorMessage('Transform failed: ' + data.message);
      }
    } catch {
      setErrorMessage('Could not connect to backend. Make sure it is running on port 8000.');
    } finally {
      setIsLoading(false);
    }
  };

  // ── Render ────────────────────────────────────────────────────────
  return (
    <ThemeProvider>
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 transition-colors">
        <div className="max-w-[1400px] mx-auto p-6">
          <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl overflow-hidden transition-colors">
            <Header />

            {/* Global error banner */}
            {errorMessage && (
              <div className="mx-6 mt-4 px-4 py-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg flex items-center justify-between">
                <p className="text-red-600 dark:text-red-400 text-sm">{errorMessage}</p>
                <button
                  onClick={() => setErrorMessage(null)}
                  className="text-red-400 hover:text-red-600 dark:hover:text-red-300 ml-4 text-lg leading-none"
                >
                  x
                </button>
              </div>
            )}

            <div className="flex gap-6 p-6">
              <Sidebar
                selectedCategory={selectedCategory}
                onSelectCategory={handleSelectCategory}
              />

              {/* PreviewArea shows transformed image if available, otherwise original */}
              <PreviewArea
                uploadedImage={previewUrl ?? uploadedImage}
                onUploadImage={handleUploadImage}
                intensity={intensity}
                isLoading={isLoading}
              />

              <ControlPanel
                selectedCategory={selectedCategory}
                selectedTreatment={selectedTreatment}
                onTreatmentChange={setSelectedTreatment}
                intensity={intensity}
                onIntensityChange={setIntensity}
                xPosition={xPosition}
                onXPositionChange={setXPosition}
                yPosition={yPosition}
                onYPositionChange={setYPosition}
                onPreview={handlePreview}
                onApplyFinal={handleApplyFinal}
              />
            </div>
          </div>
        </div>
      </div>
    </ThemeProvider>
  );
}