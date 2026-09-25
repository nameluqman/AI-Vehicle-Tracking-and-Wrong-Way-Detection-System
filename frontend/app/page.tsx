'use client';

import React, { useState, useEffect } from 'react';

const API_BASE_URL = 'http://localhost:8000';

interface VideoState {
  video_id: string;
  original_filename: string;
  status: string;
  violations_count: number;
}

interface Violation {
  id: string;
  video_id: string;
  track_id: number;
  filename: string;
  image_url: string;
  timestamp_frame: number;
  violation_type: string;
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [currentVideo, setCurrentVideo] = useState<VideoState | null>(null);
  const [violations, setViolations] = useState<Violation[]>([]);
  const [isUploading, setIsUploading] = useState(false);

  // Poll video status & violations when active
  useEffect(() => {
    if (!currentVideo?.video_id) return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/v1/videos/${currentVideo.video_id}`);
        if (res.ok) {
          const data: VideoState = await res.json();
          setCurrentVideo(data);
        }

        const vRes = await fetch(`${API_BASE_URL}/api/v1/violations?video_id=${currentVideo.video_id}`);
        if (vRes.ok) {
          const vData: Violation[] = await vRes.json();
          setViolations(vData);
        }
      } catch (err) {
        console.error('Error fetching status updates:', err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [currentVideo?.video_id]);

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/videos/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('Failed to upload file.');

      const data: VideoState = await res.json();
      setCurrentVideo(data);
      setViolations([]);
    } catch (err) {
      alert('Upload failed: ' + (err as Error).message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-8 font-sans">
      <header className="mb-8 border-b border-slate-800 pb-4">
        <h1 className="text-3xl font-bold text-indigo-400">AI Vehicle Tracking and Wrong-Way Detection System</h1>
        <p className="text-slate-400 text-sm mt-1">Real-Time Live YOLO ByteTrack Vehicle Tracking and Wrong-Way Analytics</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Upload Controls & Status */}
        <div className="space-y-6">
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-lg">
            <h2 className="text-xl font-semibold mb-4 text-slate-200">Upload Traffic Video</h2>
            <form onSubmit={handleUpload} className="space-y-4">
              <input
                type="file"
                accept="video/*"
                onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                className="block w-full text-sm text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-500 cursor-pointer"
              />
              <button
                type="submit"
                disabled={!selectedFile || isUploading}
                className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 text-white font-medium rounded-md transition disabled:opacity-50"
              >
                {isUploading ? 'Uploading...' : 'Upload & Stream Feed'}
              </button>
            </form>
          </div>

          {currentVideo && (
            <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-lg space-y-3">
              <h3 className="text-lg font-semibold text-slate-200">Video Information</h3>
              <p className="text-sm text-slate-400">ID: <span className="text-slate-200">{currentVideo.video_id}</span></p>
              <p className="text-sm text-slate-400">Status: <span className="font-semibold text-indigo-400">{currentVideo.status}</span></p>
              <p className="text-sm text-slate-400">Violations Flagged: <span className="font-semibold text-rose-400">{violations.length}</span></p>
            </div>
          )}
        </div>

        {/* Center Column: Live Video Stream */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-lg">
            <h2 className="text-xl font-semibold mb-4 text-slate-200">
              Real-Time Tracking Feed
            </h2>

            {currentVideo ? (
              <div className="video-player-wrapper border border-slate-700 rounded-lg">
                <img
                  src={`${API_BASE_URL}/api/v1/videos/stream/${currentVideo.video_id}`}
                  alt="Real-Time Tracking Stream"
                  className="stream-media rounded-lg"
                />
              </div>
            ) : (
              <div className="h-80 flex items-center justify-center text-slate-500 border border-dashed border-slate-700 rounded-lg">
                Upload a video to display live tracking.
              </div>
            )}
          </div>

          {/* Bottom Section: Captured Violation Snapshots */}
          {violations.length > 0 && (
            <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-lg">
              <h2 className="text-xl font-semibold mb-4 text-rose-400">Captured Violations & ReID Indexes</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {violations.map((v) => (
                  <div key={v.id} className="bg-slate-900 border border-slate-700 rounded-lg p-2 space-y-2">
                    <img
                      src={`${API_BASE_URL}${v.image_url}`}
                      alt={`Track ${v.track_id}`}
                      className="w-full h-28 object-cover rounded"
                    />
                    <div className="text-xs">
                      <p className="text-rose-400 font-bold">{v.violation_type}</p>
                      <p className="text-slate-400">Track ID: #{v.track_id}</p>
                      <p className="text-slate-500">Frame: {v.timestamp_frame}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}