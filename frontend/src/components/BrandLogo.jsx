import React from 'react';

/**
 * Official Vox ALM Soundwave Microphone Brand Logo Component
 */
export function BrandLogo({ size = 'w-7 h-7', className = '', showText = false, textClassName = '' }) {
  return (
    <div className={`flex items-center gap-2.5 shrink-0 ${className}`}>
      <div className={`${size} rounded-lg bg-[#181818] border border-[#333333] p-1 flex items-center justify-center shrink-0 shadow-md overflow-hidden group`}>
        <img
          src="/logo.png"
          alt="Vox ALM Logo"
          className="w-full h-full object-contain filter invert contrast-125 group-hover:scale-105 transition-transform"
        />
      </div>
      {showText && (
        <div className="flex flex-col min-w-0">
          <span className={`text-sm font-bold tracking-tight text-[#ececec] truncate ${textClassName}`}>
            Vox <span className="text-amber-400 font-mono text-xs ml-0.5">ALM</span>
          </span>
          <span className="text-[10px] text-[#8e8ea0] tracking-wide truncate">
            Audio Language Model
          </span>
        </div>
      )}
    </div>
  );
}

export default BrandLogo;
