'use client';

import React, { useState, useEffect, useRef } from 'react';
import LaneCalibration from '../components/LaneCalibration';
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
  const [isCalibrated, setIsCalibrated] = useState(false);
  const [violations, setViolations] = useState<Violation[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  // Connect to WebSocket for real-time violation updates when calibrated
  useEffect(() => {
    if (!currentVideo?.video_id || !isCalibrated) {
      // Disconnect WebSocket if not calibrated
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      return;
    }

    let cancelled = false;
    fetch(`${API_BASE_URL}/api/v1/violations?video_id=${currentVideo.video_id}`)
      .then(async (res) => {
        if (!res.ok) throw new Error('Failed to fetch violations.');
        return res.json() as Promise<Violation[]>;
      })
      .then((existingViolations) => {
        if (!cancelled) {
          setViolations((previous) => {
            const byId = new Map(previous.map((violation) => [violation.id, violation]));
            existingViolations.forEach((violation) => byId.set(violation.id, violation));
            return [...byId.values()].sort((a, b) => b.timestamp_frame - a.timestamp_frame);
          });
        }
      })
      .catch((err) => console.error('Error fetching violations:', err));

    // Connect to WebSocket
    const wsUrl = `ws://localhost:8000/ws/violations/${currentVideo.video_id}`;
    console.log('Connecting to WebSocket:', wsUrl);
    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onopen = () => {
      console.log('WebSocket connected for real-time violations');
    };

    wsRef.current.onmessage = (event) => {
      try {
        const violation: Violation = JSON.parse(event.data);
        console.log('New violation detected:', violation);
        // Add new violation to the list (prepend to show newest first)
        setViolations(prev => {
          // Check if this violation already exists to avoid duplicates
          const exists = prev.some(v => v.id === violation.id);
          if (exists) return prev;
          return [violation, ...prev];
        });
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };

    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    wsRef.current.onclose = () => {
      console.log('WebSocket disconnected');
    };

    // Cleanup on unmount or when video changes
    return () => {
      cancelled = true;
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [currentVideo?.video_id, isCalibrated]);

  // Poll video status (violations now come via WebSocket)
  useEffect(() => {
    if (!currentVideo?.video_id || !isCalibrated) return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/v1/videos/${currentVideo.video_id}`);
        if (res.ok) {
          const data: VideoState = await res.json();
          setCurrentVideo(data);
        }
      } catch (err) {
        console.error('Error fetching status updates:', err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [currentVideo?.video_id, isCalibrated]);

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
      setIsCalibrated(false); // Require calibration for newly uploaded video
      setViolations([]);
    } catch (err) {
      alert('Upload failed: ' + (err as Error).message);
    } finally {
      setIsUploading(false);
    }
  };

  const resetFlow = () => {
    // Close WebSocket connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setCurrentVideo(null);
    setIsCalibrated(false);
    setViolations([]);
    setSelectedFile(null);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-8 font-sans">
      <header className="mb-8 border-b border-slate-800 pb-4 flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-indigo-400">AI Vehicle Tracking and Wrong-Way Detection System</h1>
          <p className="text-slate-400 text-sm mt-1">Real-Time Live YOLO ByteTrack Vehicle Tracking and Wrong-Way Analytics</p>
        </div>
        {currentVideo && (
          <button
            onClick={resetFlow}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm font-medium rounded-lg border border-slate-700 transition"
          >
            Upload New Video
          </button>
        )}
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Upload Controls & Video Information */}
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
                {isUploading ? 'Uploading...' : 'Upload & Proceed to Calibration'}
              </button>
            </form>
          </div>

          {currentVideo && (
            <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-lg space-y-3">
              <h3 className="text-lg font-semibold text-slate-200">Video Information</h3>
              <p className="text-sm text-slate-400">ID: <span className="text-slate-200">{currentVideo.video_id}</span></p>
              <p className="text-sm text-slate-400">
                Calibration Status:{' '}
                <span className={`font-semibold ${isCalibrated ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {isCalibrated ? 'Calibrated' : 'Pending Screen Line Selection'}
                </span>
              </p>
              <p className="text-sm text-slate-400">Stream Status: <span className="font-semibold text-indigo-400">{currentVideo.status}</span></p>
              <p className="text-sm text-slate-400">Violations Flagged: <span className="font-semibold text-rose-400">{violations.length}</span></p>
            </div>
          )}
        </div>

        {/* Center/Right Area: Calibration OR Live Stream + Violations */}
        <div className="lg:col-span-2 space-y-6">
          {!currentVideo && (
            <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-lg h-96 flex items-center justify-center text-slate-500 border-dashed">
              Upload a video from the panel on the left to begin setup.
            </div>
          )}

          {/* Step 1: Canvas Line Drawing Calibration */}
          {currentVideo && !isCalibrated && (
            <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-lg">
              <h2 className="text-xl font-semibold mb-4 text-slate-200">Lane Calibration Setup</h2>
              <LaneCalibration
                videoId={currentVideo.video_id}
                onCalibrationComplete={() => setIsCalibrated(true)}
              />
            </div>
          )}

          {/* Step 2: Live Video Stream Feed */}
          {currentVideo && isCalibrated && (
            <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-lg">
              <h2 className="text-xl font-semibold mb-4 text-slate-200">
                Real-Time Tracking Feed
              </h2>
              <div className="video-player-wrapper border border-slate-700 rounded-lg overflow-hidden">
                <img
                  src={`${API_BASE_URL}/api/v1/videos/stream/${currentVideo.video_id}`}
                  alt="Real-Time Tracking Stream"
                  className="stream-media w-full rounded-lg"
                />
              </div>
            </div>
          )}

          {/* Step 3: Captured Violation Snapshots */}
          {currentVideo && isCalibrated && (
            <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-lg">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold text-rose-400">Captured Violations & ReID Indexes</h2>
                <span className="bg-rose-600 text-white px-3 py-1 rounded-full text-sm font-bold">
                  {violations.length} {violations.length === 1 ? 'Violation' : 'Violations'}
                </span>
              </div>
              {violations.length > 0 ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {violations.map((v) => (
                    <div key={v.id} className="bg-slate-900 border border-slate-700 rounded-lg p-2 space-y-2 animate-pulse-once">
                      <img
                        src={`${API_BASE_URL}${v.image_url}`}
                        alt={`Track ${v.track_id}`}
                        className="w-full h-28 object-cover rounded border-2 border-rose-500"
                      />
                      <div className="text-xs">
                        <p className="text-rose-400 font-bold">{v.violation_type}</p>
                        <p className="text-slate-400">Track ID: #{v.track_id}</p>
                        <p className="text-slate-500">Frame: {v.timestamp_frame}</p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center text-slate-500 py-8">
                  <p>No violations detected yet. Monitoring in progress...</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}