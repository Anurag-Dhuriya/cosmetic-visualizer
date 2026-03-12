import { useState } from "react";

interface ControlPanelProps {
  selectedCategory: string;
  intensity: number;
  onIntensityChange: (value: number) => void;
  xPosition: number;
  onXPositionChange: (value: number) => void;
  yPosition: number;
  onYPositionChange: (value: number) => void;
}

export function ControlPanel({
  selectedCategory,
  intensity,
  onIntensityChange,
  xPosition,
  onXPositionChange,
  yPosition,
  onYPositionChange,
}: ControlPanelProps) {
  return (
    <div className="w-72 flex-shrink-0 space-y-6">
      <div>
        <h2 className="text-slate-900 dark:text-slate-100 mb-2 transition-colors">Selected Category</h2>
        <div className="px-4 py-2.5 bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-700 dark:to-slate-800 rounded-lg border border-slate-200 dark:border-slate-600 transition-colors">
          <span className="text-slate-700 dark:text-slate-300 transition-colors">{selectedCategory}</span>
        </div>
      </div>
      
      <div>
        <h2 className="text-slate-900 dark:text-slate-100 mb-2 transition-colors">Treatment</h2>
        <div className="px-4 py-2.5 bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-700 dark:to-slate-800 rounded-lg border border-slate-200 dark:border-slate-600 transition-colors">
          <span className="text-slate-700 dark:text-slate-300 transition-colors">Temples Fillers</span>
        </div>
      </div>
      
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-slate-900 dark:text-slate-100 transition-colors">Intensity</h2>
          <span className="text-slate-900 dark:text-slate-100 font-semibold text-lg transition-colors">{intensity}%</span>
        </div>
        <input
          type="range"
          min="0"
          max="100"
          value={intensity}
          onChange={(e) => onIntensityChange(Number(e.target.value))}
          className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-full appearance-none cursor-pointer transition-colors [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-gradient-to-r [&::-webkit-slider-thumb]:from-blue-500 [&::-webkit-slider-thumb]:to-blue-600 [&::-webkit-slider-thumb]:shadow-lg [&::-webkit-slider-thumb]:shadow-blue-500/30 [&::-webkit-slider-thumb]:cursor-pointer"
        />
        <p className="text-slate-500 dark:text-slate-400 text-sm mt-2 transition-colors">Use the slider to control strength (0–100%)</p>
      </div>
      
      <div>
        <h2 className="text-slate-900 dark:text-slate-100 mb-3 transition-colors">Position Adjust</h2>
        <div className="space-y-4">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-600 dark:text-slate-400 text-sm transition-colors">X (left/right)</span>
              <span className="text-slate-900 dark:text-slate-100 font-medium transition-colors">{xPosition}</span>
            </div>
            <input
              type="range"
              min="-50"
              max="50"
              value={xPosition}
              onChange={(e) => onXPositionChange(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-full appearance-none cursor-pointer transition-colors [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-gradient-to-r [&::-webkit-slider-thumb]:from-slate-400 [&::-webkit-slider-thumb]:to-slate-500 [&::-webkit-slider-thumb]:cursor-pointer"
            />
          </div>
          
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-600 dark:text-slate-400 text-sm transition-colors">Y (up/down)</span>
              <span className="text-slate-900 dark:text-slate-100 font-medium transition-colors">{yPosition}</span>
            </div>
            <input
              type="range"
              min="-50"
              max="50"
              value={yPosition}
              onChange={(e) => onYPositionChange(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-full appearance-none cursor-pointer transition-colors [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-gradient-to-r [&::-webkit-slider-thumb]:from-slate-400 [&::-webkit-slider-thumb]:to-slate-500 [&::-webkit-slider-thumb]:cursor-pointer"
            />
          </div>
        </div>
      </div>
      
      <div>
        <h2 className="text-slate-900 dark:text-slate-100 mb-3 transition-colors">Actions</h2>
        <div className="flex gap-2">
          <button onClick={useState}
            
           className="flex-1 px-4 py-2.5 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg hover:from-blue-600 hover:to-blue-700 transition-all shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/40  ">
            Preview (fast)
          </button>
          <button className="flex-1 px-4 py-2.5 bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-600 transition-all border border-slate-200 dark:border-slate-600">
            Apply Final
          </button>
        </div>
      </div>
      
      <div>
        <h2 className="text-slate-900 dark:text-slate-100 mb-2 transition-colors">Notes</h2>
        <div className="bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-4 transition-colors">
          <p className="text-slate-600 dark:text-slate-300 text-sm leading-relaxed transition-colors">
            Preview uses a low-res copy for speed. Apply final to generate full-resolution result.
          </p>
        </div>
      </div>
    </div>
  );
}