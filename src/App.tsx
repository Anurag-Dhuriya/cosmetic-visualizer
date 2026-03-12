import { useState } from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { PreviewArea } from './components/PreviewArea';
import { ControlPanel } from './components/ControlPanel';
import { ThemeProvider } from './contexts/ThemeContext';

export default function App() {
  const [selectedCategory, setSelectedCategory] = useState('Face');
  const [intensity, setIntensity] = useState(60);
  const [xPosition, setXPosition] = useState(0);
  const [yPosition, setYPosition] = useState(0);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);

  return (
    <ThemeProvider>
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 transition-colors">
        <div className="max-w-[1400px] mx-auto p-6">
          <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl overflow-hidden transition-colors">
            <Header />
            
            <div className="flex gap-6 p-6">
              <Sidebar 
                selectedCategory={selectedCategory}
                onSelectCategory={setSelectedCategory}
              />
              
              <PreviewArea 
                uploadedImage={uploadedImage}
                onUploadImage={setUploadedImage}
                intensity={intensity}
              />
              
              <ControlPanel 
                selectedCategory={selectedCategory}
                intensity={intensity}
                onIntensityChange={setIntensity}
                xPosition={xPosition}
                onXPositionChange={setXPosition}
                yPosition={yPosition}
                onYPositionChange={setYPosition}
              />
            </div>
          </div>
        </div>
      </div>
    </ThemeProvider>
  );
}