import { Upload } from 'lucide-react';

interface PreviewAreaProps {
  uploadedImage: string | null;
  onUploadImage: (image: string | null) => void;
  intensity: number;
}

export function PreviewArea({ uploadedImage, onUploadImage, intensity }: PreviewAreaProps) {
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        onUploadImage(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="flex-1 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-slate-900 dark:text-slate-100 transition-colors">Upload patient photo</h2>
          <p className="text-slate-500 dark:text-slate-400 text-sm transition-colors">Accepted: JPG, PNG • Max: 10 MB</p>
        </div>
        <label className="cursor-pointer">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />
          <div className="px-4 py-2.5 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-600 transition-all text-slate-700 dark:text-slate-300 hover:border-slate-300 dark:hover:border-slate-500 inline-flex items-center gap-2">
            <Upload className="w-4 h-4" />
            <span>Upload Photo</span>
          </div>
        </label>
      </div>
      
      <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden shadow-sm transition-colors">
        <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-slate-50 to-white dark:from-slate-800 dark:to-slate-800 border-b border-slate-200 dark:border-slate-700 transition-colors">
          <div className="flex items-center gap-6">
            <h3 className="text-blue-600 dark:text-blue-400 transition-colors">Face</h3>
            <span className="text-slate-500 dark:text-slate-400 text-sm transition-colors">image_id: demo-0001</span>
          </div>
          <div className="flex items-center gap-4 text-sm">
            <span className="text-slate-600 dark:text-slate-400 transition-colors">Before</span>
            <span className="text-slate-600 dark:text-slate-400 transition-colors">After (Preview)</span>
          </div>
        </div>
        
        <div className="aspect-[16/10] bg-gradient-to-br from-slate-800 to-slate-900 dark:from-slate-900 dark:to-black flex items-center justify-center relative transition-colors">
          {uploadedImage ? (
            <img src={uploadedImage} alt="Preview" className="w-full h-full object-cover" />
          ) : (
            <div className="text-center">
              <Upload className="w-12 h-12 text-slate-600 dark:text-slate-500 mx-auto mb-3 transition-colors" />
              <p className="text-slate-500 dark:text-slate-400 transition-colors">Upload a photo to preview</p>
            </div>
          )}
        </div>
        
        <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-slate-50 to-white dark:from-slate-800 dark:to-slate-800 border-t border-slate-200 dark:border-slate-700 transition-colors">
          <p className="text-slate-500 dark:text-slate-400 text-sm transition-colors">Live preview updates while you move sliders</p>
          <div className="flex items-center gap-2">
            <span className="text-slate-600 dark:text-slate-400 text-sm transition-colors">Intensity</span>
            <span className="text-slate-900 dark:text-slate-100 font-medium transition-colors">{intensity}%</span>
          </div>
        </div>
      </div>
      
      <p className="text-center text-slate-400 dark:text-slate-500 text-sm transition-colors">
        Demo UI • Not a medical device • For visualization only
      </p>
    </div>
  );
}