"use client";

import React, { useRef, useState, useEffect, useCallback } from "react";

interface Point {
  x: number;
  y: number;
}

interface LaneCalibrationProps {
  videoId: string;
  onCalibrationComplete: () => void;
}

export default function LaneCalibration({ videoId, onCalibrationComplete }: LaneCalibrationProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);

  const [upwardPoints, setUpwardPoints] = useState<Point[]>([]);
  const [downwardPoints, setDownwardPoints] = useState<Point[]>([]);
  const [stage, setStage] = useState<"upward" | "downward" | "complete">("upward");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const API_BASE = "http://localhost:8000";

  // Target standard frame size used by pipeline.py
  const TARGET_WIDTH = 1280;
  const TARGET_HEIGHT = 720;

  // 1. Fetch First Frame Image
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = `${API_BASE}/api/v1/videos/first-frame/${videoId}`;
    img.onload = () => {
      setImage(img);
      if (canvasRef.current) {
        // Force internal canvas pixel buffer to match backend stream dimensions
        canvasRef.current.width = TARGET_WIDTH;
        canvasRef.current.height = TARGET_HEIGHT;
      }
    };
  }, [videoId]);

  // Helper to draw lines with labels
  const drawLineWithLabel = (
    ctx: CanvasRenderingContext2D,
    pts: Point[],
    color: string,
    label: string
  ) => {
    ctx.fillStyle = color;
    pts.forEach((p) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 8, 0, 2 * Math.PI);
      ctx.fill();
    });

    if (pts.length === 2) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      ctx.lineTo(pts[1].x, pts[1].y);
      ctx.stroke();

      ctx.font = "bold 22px Arial";
      ctx.fillStyle = color;
      ctx.fillText(label, pts[0].x + 10, pts[0].y - 10);
    }
  };

  // 2. Draw Image scaled to 1280x720 and Calibration Lines
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Draw raw frame scaled onto 1280x720 canvas
    ctx.drawImage(image, 0, 0, TARGET_WIDTH, TARGET_HEIGHT);

    // Draw Upward Line (Green)
    if (upwardPoints.length > 0) {
      drawLineWithLabel(ctx, upwardPoints, "#00FF00", "UPWARD (GREEN)");
    }

    // Draw Downward Line (Blue)
    if (downwardPoints.length > 0) {
      drawLineWithLabel(ctx, downwardPoints, "#0088FF", "DOWNWARD (BLUE)");
    }
  }, [image, upwardPoints, downwardPoints]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  // 3. Handle Screen Clicks accurately scaled to 1280x720
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = TARGET_WIDTH / rect.width;
    const scaleY = TARGET_HEIGHT / rect.height;

    const clickPoint: Point = {
      x: Math.round((e.clientX - rect.left) * scaleX),
      y: Math.round((e.clientY - rect.top) * scaleY),
    };

    if (stage === "upward" && upwardPoints.length < 2) {
      const nextUpward = [...upwardPoints, clickPoint];
      setUpwardPoints(nextUpward);
      if (nextUpward.length === 2) {
        setStage("downward");
      }
    } else if (stage === "downward" && downwardPoints.length < 2) {
      const nextDownward = [...downwardPoints, clickPoint];
      setDownwardPoints(nextDownward);
      if (nextDownward.length === 2) {
        setStage("complete");
      }
    }
  };

  // 4. Submit Calibration Payload
  const submitCalibration = useCallback(async () => {
    if (upwardPoints.length < 2 || downwardPoints.length < 2 || isSubmitting) return;

    setIsSubmitting(true);
    const payload = {
      video_id: videoId,
      upward_points: upwardPoints.map((p) => [p.x, p.y]),
      downward_points: downwardPoints.map((p) => [p.x, p.y]),
    };

    try {
      const res = await fetch(`${API_BASE}/api/v1/videos/calibrate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        onCalibrationComplete();
      } else {
        alert("Calibration failed on backend.");
      }
    } catch (err) {
      console.error("Calibration submission error:", err);
    } finally {
      setIsSubmitting(false);
    }
  }, [upwardPoints, downwardPoints, isSubmitting, videoId, API_BASE, onCalibrationComplete]);

  // 5. Enter Key Event Listener
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Enter" && stage === "complete") {
        submitCalibration();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [stage, submitCalibration]);

  const resetCalibration = () => {
    setUpwardPoints([]);
    setDownwardPoints([]);
    setStage("upward");
  };

  return (
    <div className="flex flex-col items-center gap-4 bg-slate-900 p-6 rounded-xl border border-slate-800">
      <div className="text-sm font-semibold text-slate-300">
        {stage === "upward" && (
          <span className="text-emerald-400">Step 1: Click 2 points on screen for UPWARD lane (Green Line)</span>
        )}
        {stage === "downward" && (
          <span className="text-blue-400">Step 2: Click 2 points on screen for DOWNWARD lane (Blue Line)</span>
        )}
        {stage === "complete" && (
          <span className="text-amber-400">Step 3: Calibration Ready! Press ENTER or click Submit to Start Streaming.</span>
        )}
      </div>

      <div className="relative border-2 border-slate-700 rounded-lg overflow-hidden w-full max-w-4xl aspect-video bg-black flex items-center justify-center">
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          className="cursor-crosshair w-full h-full object-contain block"
        />
      </div>

      <div className="flex gap-4">
        <button
          onClick={resetCalibration}
          className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm font-medium rounded-lg transition"
        >
          Reset Lines
        </button>

        <button
          onClick={submitCalibration}
          disabled={stage !== "complete" || isSubmitting}
          className={`px-6 py-2 text-sm font-medium rounded-lg transition ${
            stage === "complete" && !isSubmitting
              ? "bg-indigo-600 hover:bg-indigo-500 text-white cursor-pointer"
              : "bg-slate-800 text-slate-500 cursor-not-allowed"
          }`}
        >
          {isSubmitting ? "Starting Feed..." : "Submit & Start Feed (Enter)"}
        </button>
      </div>
    </div>
  );
}