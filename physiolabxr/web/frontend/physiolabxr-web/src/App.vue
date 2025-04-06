<template>
  <div class="app">
    <!-- Hidden file input (not visible, only triggered by the button) -->
    <input
      type="file"
      ref="fileInput"
      style="display: none;"
      @change="onFileSelected"
    />

    <!-- Button that opens the file explorer -->
    <button :disabled="isUploading" @click="openFileDialog">
      {{ isUploading ? 'Uploading...' : 'Load Replay' }}
    </button>

    <!-- If no frames yet, show a message -->
    <div v-if="frameCount === 0">
      <p>No plx recording loaded. Please upload a file.</p>
    </div>


    <!-- Otherwise, show the manual video UI -->
    <div v-else>
      <!-- Main frame display -->
<!--      <div class="preview">-->
<!--        <img :src="getFrameUrl(currentFrameIndex)" alt="Current Frame" />-->
<!--      </div>-->

      <div
        class="preview"
        :style="{
          width: previewWidth + 'px',
          height: previewHeight + 'px'
        }"
      >
        <img :src="getFrameUrl(currentFrameIndex)" alt="Current Frame" />
        <canvas ref="traceCanvas" class="gaze-trace"></canvas>

        <!-- The gaze dot on top -->
        <div v-if="showGaze" class="gaze-dot" :style="gazeDotStyle"></div>
      </div>

      <!-- Playback controls -->
      <div class="controls">
        <button @click="togglePlay">{{ isPlaying ? "Pause" : "Play" }}</button>
        <button @click="stopPlayback">Stop</button>
        <span class="time-display">
          {{ currentVideoTimeFormatted }} / {{ totalTimeFormatted }}
        </span>

        <!-- Wrap the frame display -->
        <span class="time-display">
          {{ currentFrameIndex }} / {{ frameCount }}
        </span>
<!--        <span class="time-display">-->
<!--          {{ currentTimestampFormatted }}-->
<!--        </span>-->
      </div>

      <!-- Timeline with thumbnails -->
      <!-- Timeline -->
      <div class="tracks">
        <div
          class="timeline"
          ref="timelineEl"
          tabindex="0"
          @wheel="onTimelineWheel"
          @scroll="onTimelineScroll"
          @click="onTimelineClick"
          @keydown.stop.prevent="onTimelineKeydown"
        >
          <div
            class="timeline-expandable"
            ref="timelineExpandable"
            :style="{ width: totalTimelineWidth + 'px' }"
          >
            <div
              v-for="(thumb, i) in filteredThumbnails"
              :key="i"
              class="thumbnail"
              :style="{ left: (i*100) + 'px' }"
            >
              <img :src="getFrameUrl(thumb.index)" alt="..." />
              <div>{{ formatTimestamp(thumb.timestamp) }}</div>
            </div>

            <!-- Playhead -->
            <div class="playhead" :style="{ left: playheadX + 'px' }"></div>
          </div>
        </div>
        <!-- The Zoom Overlay in bottom-right corner -->
        <div class="zoom-overlay" v-if="showZoomOverlay">
          Zoom: {{ zoomLevel.toFixed(2) }}
        </div>
      </div>

      <!-- Chart.js pupil data -->
      <div class="chart-inner" ref="chartContainer">
          <canvas ref="chartCanvas"></canvas>
      </div>
    </div>

  </div>
</template>

<script>
import { ref, onMounted, computed, watch, nextTick} from "vue";
import Chart from "chart.js/auto";
import annotationPlugin from "chartjs-plugin-annotation";
Chart.register(annotationPlugin);

export default {
  name: "App",
  setup() {
    /**
     * -------------------------------------------
     * 1) File Upload logic
     * -------------------------------------------
     */
    const fileInput = ref(null);
    const isUploading = ref(false);

    const openFileDialog = () => {
      fileInput.value.click();
    };

    const onFileSelected = async (event) => {
      const files = event.target.files;
      if (!files || !files.length) return;

      // We only handle one file for this example
      const file = files[0];
      console.log("Selected file:", file.name);
      isUploading.value = true;

      // Create FormData for file upload
      const formData = new FormData();
      formData.append("replay_file", file);

      try {
        // POST to your backend to handle the replay file
        const res = await fetch("/api/upload_replay", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();

        if (!res.ok) {
          alert(data.error || "Upload error");
          return;
        }

        console.log("File uploaded successfully:", data);

        // After uploading, fetch the new video info
        await fetchVideoInfo();

        // Optionally also fetch gaze data (if relevant)
        await fetchGazeAligned();

        await fetchPupilData();

        renderPupilChart();
      } catch (err) {
        console.error("Upload failed:", err);
        alert("Failed to upload file");
      } finally {
        isUploading.value = false;
      }
    };

    onMounted(async () => {
      await fetchVideoInfo();
      await fetchGazeAligned();
      await fetchPupilData();
      renderPupilChart();
      updateChartViewport();
    });

    /**
     * -------------------------------------------
     * 2) Manual Video Playback logic
     * -------------------------------------------
     */
    const frameCount = ref(0);
    const total_duration_sec = ref(0);
    const timestamps = ref([]); // array of float timestamps
    const currentFrameIndex = ref(0);
    const average_fps = ref(0);
    const average_frame_interval_ms = ref(0);
    const video_resolution = ref([]);
    const isPlaying = ref(false);
    let playInterval = null;

    const fetchVideoInfo = async () => {
      try {
        const res = await fetch("/api/video_info");
        const data = await res.json();
        if (data.error) {
          console.warn("No video loaded or error:", data.error);
          frameCount.value = 0;
          timestamps.value = [];
          total_duration_sec.value = 0;
          average_fps.value = 0;
          average_frame_interval_ms.value = 0;
          video_resolution.value = [];
          return;
        }
        frameCount.value = data.frame_count;
        timestamps.value = data.timestamps;
        total_duration_sec.value = data.total_duration_sec;
        average_fps.value = data.average_fps;
        average_frame_interval_ms.value = parseInt(1000 / data.average_fps);
        video_resolution.value = data.video_resolution;
        currentFrameIndex.value = 0;
      } catch (err) {
        console.error("fetchVideoInfo failed:", err);
        frameCount.value = 0;
        timestamps.value = [];
        total_duration_sec.value = 0;
        average_fps.value = 0;
        average_frame_interval_ms.value = 0;
        video_resolution.value = [];
      }
    };

    // Return the URL for a given frame index
    const getFrameUrl = (idx) => {
      return `/api/frame/${idx}`;
    };

    // Basic play/pause
    const togglePlay = () => {
      if (isPlaying.value) {
        // pause
        isPlaying.value = false;
        clearInterval(playInterval);
        centerPlayheadInTimeline();
      } else {
        // play
        isPlaying.value = true;
        playInterval = setInterval(() => {
          if (currentFrameIndex.value < frameCount.value - 1) {
            currentFrameIndex.value++;
          } else {
            stopPlayback();
          }
        }, average_frame_interval_ms.value); // ~10fps "playback"
      }
    };

    const stopPlayback = () => {
      isPlaying.value = false;
      clearInterval(playInterval);
      currentFrameIndex.value = 0;

      // Clear the trace, because continuity is lost
      clearGazeHistory();
    };
    const timelineEl = ref(null);

    // Format timestamps (seconds) as mm:ss
    const formatTimestamp = (sec) => {
      const m = Math.floor(sec / 60);
      const s = Math.floor(sec % 60);
      const ms = Math.floor((sec % 1) * 1000);
      return `${m}:${s < 10 ? "0" : ""}${s}.${ms}`;
    };

    // The current VideoTime is the current timestamp - first timestamp
    const currentVideoTimeFormatted = computed(() => {
      if (!timestamps.value || timestamps.value.length === 0) {
        return "0:00";
      }
      const t = timestamps.value[currentFrameIndex.value] - timestamps.value[0] || 0;
      return formatTimestamp(t);
    });

    const currentTimestampFormatted = computed(() => {
      if (!timestamps.value || timestamps.value.length === 0) {
        return "0:00";
      }
      const t = timestamps.value[currentFrameIndex.value] || 0;
      return formatTimestamp(t);
    });

    const totalTimeFormatted = computed(() => {
      if (!timestamps.value || timestamps.value.length === 0) {
        return "0:00";
      }
      return formatTimestamp(total_duration_sec.value);
    });

    // timeline zoom logic
    const zoomLevel = ref(1); // 1 = fully zoomed out, bigger = more zoomed in
    const MIN_ZOOM = 1;
    const MAX_ZOOM = 10;

        // The function that handles pinch or ctrl+wheel:
    const showZoomOverlay = ref(false);
    let overlayTimer = null;

    const onTimelineWheel = async (e) => {
      if (e.ctrlKey) {
        e.preventDefault();
        let delta = e.deltaY * 0.01; // sensitivity
        zoomLevel.value += delta;
        if (zoomLevel.value < MIN_ZOOM) zoomLevel.value = MIN_ZOOM;
        if (zoomLevel.value > MAX_ZOOM) zoomLevel.value = MAX_ZOOM;

        // Show the overlay
        showZoomOverlay.value = true;

        // Clear any existing timer
        if (overlayTimer) {
          clearTimeout(overlayTimer);
        }
        // Start a new 1-second timer
        overlayTimer = setTimeout(() => {
          showZoomOverlay.value = false;
        }, 1000);

        await nextTick();

        centerPlayheadInTimeline();
      }
      else {}
    };

    function onTimelineScroll() {
      updateChartViewport();
    }

    // 2) skip factor so we end up with ~ displayedCount frames
    const filteredThumbnails = computed(() => {
      const n = frameCount.value;
      if (n === 0) return [];
      const want = displayedCount.value;
      if (want <= 1) return [];

      const skip = Math.floor(n / (want-1));
      const arr = [];
      let i = 0;
      while (i < n) {
        arr.push({
          index: i,
          timestamp: timestamps.value[i]
        });
        i += skip;
      }
      // ensure last frame is included if not already
      if (arr[arr.length - 1].index !== n - 1) {
        arr[arr.length - 1] = {
          index: n - 1,
          timestamp: timestamps.value[n - 1]
        };
      }
      return arr;
    });

    // 1) Lerp from frameCount (zoom=1) down to 8 (zoom=10)
    // e.g. displayedCount(1) = frameCount, displayedCount(10) = 8
    const displayedCount = computed(() => {
      if (frameCount.value === 0) return 0;
      const z = zoomLevel.value;
      // fraction from 0..1
      const frac = (z - 1) / (10 - 1); // 0 at z=1, 1 at z=10
      // linear interpolation
      const startVal = frameCount.value; // at zoom=1
      const endVal = 8;                 // at zoom=10
      const val = startVal + (endVal - startVal) * frac;
      return Math.floor(val);
    });

    // 3) timelineWidth = displayedCount * 100
    const totalTimelineWidth = computed(() => {
      return Math.max(0, displayedCount.value - 1) * 100;
    });

    // 4) playhead => fraction of entire video * (displayedCount-1)*100
    const playheadX = computed(() => {
      if (frameCount.value < 2 || displayedCount.value < 2) return 0;
      const fraction = currentFrameIndex.value / (frameCount.value - 1);
      return fraction * ((displayedCount.value - 1) * 100);
    });

    // Center the playhead in the timeline if possible
  const centerPlayheadInTimeline = () => {
    // Access the DOM element
    const timeline = timelineEl.value;
    if (!timeline) return;

    // How many pixels from the left is the playhead?
    // (playheadX is a computed that returns the left position in px)
    const playheadLeftPx = playheadX.value;

    // We want the playhead to appear in the horizontal center of the timeline.
    // That center is timeline.clientWidth / 2 (half its visible width).
    const halfVisibleWidth = timeline.clientWidth / 2;

    // So if the playhead is at X, we want scrollLeft to be (X - halfVisibleWidth).
    // That tries to push the timeline so the playhead is in the middle.
    let targetScroll = playheadLeftPx - halfVisibleWidth;

    // Clamp so we don’t scroll past the left or right extremes
    const maxScroll = timeline.scrollWidth - timeline.clientWidth;
    if (targetScroll < 0) targetScroll = 0;
    if (targetScroll > maxScroll) targetScroll = maxScroll;

    // Jump immediately (or use `behavior: 'smooth'` if you want an animation)
    timeline.scrollTo({ left: targetScroll });
  };

  const onTimelineClick = (e) => {
    // 1) We find the local X coordinate in the timeline
    //    taking into account the bounding rect and the horizontal scroll
    const timeline = timelineEl.value;
    if (!timeline) return;

    // The bounding box of .timeline in page coordinates
    const rect = timeline.getBoundingClientRect();

    // e.clientX is the mouse position in page coords
    // So localX is how far from the left edge of .timeline
    // But we also must add `timeline.scrollLeft` if the user scrolled inside it
    const localX = e.clientX - rect.left + timeline.scrollLeft;

    // 2) Convert localX to a fraction of the total timeline width
    // If totalTimelineWidth is something like (displayedCount - 1) * 100
    // we do:
    const maxWidth = totalTimelineWidth.value; // e.g. 4900 if 50 frames displayed
    if (maxWidth <= 0) return;
    let fraction = localX / maxWidth;
    if (fraction < 0) fraction = 0;
    if (fraction > 1) fraction = 1;

    // 3) Convert fraction to a frame index in [0..frameCount-1]
    const newFrame = Math.round(fraction * (frameCount.value - 1));

    // 4) Set currentFrameIndex
    currentFrameIndex.value = newFrame;
    clearGazeHistory();
  };

  const onTimelineKeydown = (evt) => {
    if (frameCount.value < 1) return;

    if (evt.key === "ArrowLeft") {
      // Move back one frame
      if (currentFrameIndex.value > 0) {
        currentFrameIndex.value--;
      }
    } else if (evt.key === "ArrowRight") {
      // Move forward one frame
      if (currentFrameIndex.value < frameCount.value - 1) {
        currentFrameIndex.value++;
      }
    }
  };
  /**
   * -------------------------------------------
   * 3) (Optional) Gaze Data + Chart.js
   * -------------------------------------------
   */
  const previewWidth = ref(640);
  const previewHeight = ref(360);
  const gazeXaligned = ref([]);
  const gazeYaligned = ref([]);

  async function fetchGazeAligned() {
    const res = await fetch("/api/gaze_aligned");
    const data = await res.json();
    if (data.error) {
      console.error("No aligned gaze data or error", data.error);
      return;
    }
    gazeXaligned.value = data.x_aligned || [];
    gazeYaligned.value = data.y_aligned || [];
  }

  const showGaze = ref(true); // toggle if you want
  const gazeDotStyle = computed(() => {
    if (!frameCount.value || currentFrameIndex.value < 0) {
      return { display: "none" };
    }

    const i = currentFrameIndex.value;
    // x and y are already pixel coords
    const xVal = gazeXaligned.value[i];
    const yVal = 1 - gazeYaligned.value[i];  // minus because y=0 is top

    const dotRadius = 5;  // half of .gaze-dot's width/height
    const offsetX = (xVal * previewWidth.value) - dotRadius;
    const offsetY = (yVal * previewHeight.value) - dotRadius;

    if (xVal == null || yVal == null) {
      return {display: "none"};
    }
    return {
      left: offsetX + "px",
      top: offsetY + "px",
    };
  });

      // N frames of history
    const TRACE_LENGTH = 30; // how many frames back to show

    // We'll store the raw mapped positions in a simple array
    // each entry: { x: number, y: number }
    const gazeHistory = ref([]);

    // a ref for the canvas element
    const traceCanvas = ref(null);

    // Watch currentFrameIndex so we know each "frame" tick
    watch(currentFrameIndex, () => {
      updateGazeHistory();
      drawGazeTrace();

      const timeline = timelineEl.value;
      if (!timeline) return;

      // Left and right edges of the visible region
      const leftEdge = timeline.scrollLeft;
      const rightEdge = timeline.scrollLeft + timeline.clientWidth;

      // The playhead’s X position in timeline coordinates
      const playheadLeftPx = playheadX.value; // e.g. fraction * totalTimelineWidth

      // If the playhead is scrolled out of view,
      // then recenter (otherwise leave it alone).
      if (playheadLeftPx < leftEdge || playheadLeftPx > rightEdge) {
        centerPlayheadInTimeline();
  }
    });

    function clearGazeHistory() {
      gazeHistory.value = [];    // empty the array
      drawGazeTrace();           // also clear the canvas
    }

    function updateGazeHistory() {
      // 1) Convert the current normalized gaze coords to pixel coords
      const i = currentFrameIndex.value;
      if (i < 0 || i >= frameCount.value) return;
      const xNorm = gazeXaligned.value[i];
      const yNorm = 1 - gazeYaligned.value[i];  // if you invert y
      if (xNorm == null || yNorm == null) return;

      const xPx = xNorm * previewWidth.value;
      const yPx = yNorm * previewHeight.value;

      // 2) push into gazeHistory
      gazeHistory.value.push({ x: xPx, y: yPx });

      // 3) if we exceed TRACE_LENGTH, drop oldest
      while (gazeHistory.value.length > TRACE_LENGTH) {
        gazeHistory.value.shift();
      }
    }

  function drawGazeTrace() {
    const canvas = traceCanvas.value;
    if (!canvas) return;

    canvas.width = previewWidth.value;
    canvas.height = previewHeight.value;
    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const history = gazeHistory.value;
    if (history.length < 2) return;

    ctx.lineWidth = 2;

    for (let i = 1; i < history.length; i++) {
      const prev = history[i - 1];
      const curr = history[i];

      // Alpha decreases the further back in time (older lines = more transparent)
      const alpha = i / history.length;  // 0.0 -> 1.0
      ctx.strokeStyle = `rgba(255, 255, 0, ${alpha})`;  // yellow with fade

      ctx.beginPath();
      ctx.moveTo(prev.x, prev.y);
      ctx.lineTo(curr.x, curr.y);
      ctx.stroke();
    }
  }

    // also call draw once on mount if needed
    onMounted(() => {
      drawGazeTrace();
    });

  // Add refs to store pupil data
  const chartCanvas = ref(null);
  const chartContainer = ref(null);
  const pupilRtimes = ref([]);
  const pupilRsizes = ref([]);
  const pupilLtimes = ref([]);
  const pupilLsizes = ref([]);

  async function fetchPupilData() {
    const res = await fetch("/api/pupil_data");
    const data = await res.json();
    if (data.error) {
      console.error("No pupil data or error", data.error);
      return;
    }
    pupilRtimes.value = data.right_times;  // array of floats
    pupilRsizes.value = data.right_sizes;  // array of floats
    pupilLtimes.value = data.left_times;
    pupilLsizes.value = data.left_sizes;
  }

  let chartInstance = null; // hold reference so we can update/destroy

  function renderPupilChart() {
    if (!chartCanvas.value) return;

    // If there's an existing chart, destroy it before re-creating
    if (chartInstance) {
      chartInstance.destroy();
    }
    const ctx = chartCanvas.value.getContext("2d");

    // Convert your arrays into {x, y} for Chart.js "linear" or "time" scale
    // (If you have many data points, you might want to do this once, not on every render)
    const rightData = pupilRtimes.value.map((t, i) => ({ x: t, y: pupilRsizes.value[i] }));
    const leftData  = pupilLtimes.value.map((t, i) => ({ x: t, y: pupilLsizes.value[i] }));

    chartInstance = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          {
            label: "Right Pupil (mm)",
            data: rightData,
            borderColor: "red",
            fill: false,
          },
          {
            label: "Left Pupil (mm)",
            data: leftData,
            borderColor: "blue",
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: "linear",   // or "time" if you prefer Chart.js time scale
            min: 0,
            max: total_duration_sec.value, // same as your replay's total time
            title: { display: true, text: "Time (seconds)" },
          },
          y: {
            title: { display: true, text: "Pupil Size (mm)" },
          },
        },
        // We'll add an annotation for the "playhead" below
        plugins: {
          annotation: {
            annotations: {
              currentLine: {
                type: "line",
                xMin: 0,     // we’ll update dynamically
                xMax: 0,
                borderColor: "yellow",
                borderWidth: 2,
              }
            }
          }
        }
      },
      plugins: [Chart.registry.getPlugin('annotation')] // if needed
    });
  }

  const timelineExpandable = ref(null);

  function computeTimelineScale() {
    // The total timeline width in pixels
    const expand = timelineExpandable.value;
    if (!expand) return 0;

    const totalWidthPx = expand.offsetWidth + 150; // e.g. maybe 5000 px
    const totalSec = total_duration_sec.value; // from your data
    if (totalWidthPx <= 0 || totalSec <= 0) return 0;
    return totalSec / totalWidthPx;
  }

  function getVisibleTimeRange() {
    const timeline = timelineEl.value;
    if (!timeline) return { start: 0, end: 0 };

    const scrollLeftPx = timeline.scrollLeft;    // how many px scrolled from left
    const visibleWidthPx = timeline.clientWidth; // how many px is visible
    const secPerPx = computeTimelineScale();    // from step 2

    const startTime = scrollLeftPx * secPerPx;
    const endTime = startTime + (visibleWidthPx * secPerPx);

    return { start: startTime, end: endTime };
  }

  function updateChartViewport() {
    if (!chartInstance) return;

    const { start, end } = getVisibleTimeRange();

    chartInstance.options.scales.x.min = start;
    chartInstance.options.scales.x.max = end;

    chartInstance.update("none"); // "none" can skip animations if you like
  }

  watch(currentFrameIndex, () => {
    const idx = currentFrameIndex.value;
    if (!chartInstance) return;
    if (!timestamps.value || !timestamps.value[idx]) return;

    // The "time" in seconds for the current frame
    const currentTime = timestamps.value[idx];

    // Update the annotation’s xMin/xMax
    chartInstance.options.plugins.annotation.annotations.currentLine.xMin = currentTime;
    chartInstance.options.plugins.annotation.annotations.currentLine.xMax = currentTime;

    // Optionally, you can also "pan" or "zoom" the chart to keep currentTime in view:
    // chartInstance.options.scales.x.min = currentTime - 2; // show 2s before
    // chartInstance.options.scales.x.max = currentTime + 2; // 2s after

    chartInstance.update();
  });

  /**
   * -------------------------------------------
   * Return everything to template
   * -------------------------------------------
   */
  return {
    fileInput,
    isUploading,

    openFileDialog,
    onFileSelected,

    frameCount,
    timestamps,
    total_duration_sec,
    currentFrameIndex,
    isPlaying,
    togglePlay,
    stopPlayback,
    getFrameUrl,
    formatTimestamp,
    currentVideoTimeFormatted,
    currentTimestampFormatted,
    totalTimeFormatted,

    zoomLevel,onTimelineWheel,onTimelineScroll,onTimelineClick,showZoomOverlay,onTimelineKeydown,
    displayedCount,
    filteredThumbnails,
    totalTimelineWidth,
    timelineEl,
    playheadX,
    centerPlayheadInTimeline,

    previewWidth, previewHeight,
    showGaze,
    gazeDotStyle,
    // gazeData,
    // chartCanvas,
    traceCanvas,

    // Pupil
    chartCanvas,
    chartContainer,

    pupilRtimes,
    pupilRsizes,
    pupilLtimes,
    pupilLsizes,
    fetchPupilData,
    renderPupilChart,

    timelineExpandable,
  };
},
};
</script>

<style scoped>
.app {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: 2rem;
}

/* A black preview area for the main video frame */
.preview {
  background: black;
  position: relative; /* so absolutely positioned gaze dot is inside */
  overflow: hidden;   /* if you want to clip the dot at edges */
}
.preview img {
  max-width: 100%;
  max-height: 100%;
}

/* Controls row for play/pause, etc. */
.controls {
  width: 90%;
  max-width: 1200px;  /* or any large max you want */

  margin: 1rem;
  display: flex;
  align-items: center;
  gap: 1rem;
}

.controls button {
  /* Make sure the button won't resize between "Play" and "Pause" */
  min-width: 60px;  /* or however wide you need */
}

/* The timeline container. Horizontal scroll for many frames. */

.tracks {
  position: relative; /* The .zoom-overlay will be absolutely positioned against this container */
  overflow-x: auto;
}

.timeline-expandable {
  display: flex;
  flex-direction: column;
  width: 100%;
  height: 130px;
}

.zoom-overlay {
  position: absolute;
  right: 10px;
  bottom: 10px;
  background: rgba(0,0,0,0.6);
  color: #fff;
  padding: 4px 8px;
  border-radius: 4px;
  pointer-events: none; /* so it doesn't block clicks */
  font-size: 14px;
}

.timeline {
  position: relative;
  overflow-x: auto;
  background: #333;
  width: 80vw;      /* 80% of the viewport width */
  max-width: 1200px;  /* but never exceed 1200px, for very wide screens */
  margin-bottom: 2rem;
}

.timeline:focus {
  outline: 3px solid #66f; /* or any highlight color */
  outline-offset: 1px;   /* tweak so outline hugs the element */
}

/* Each thumbnail + timestamp label */
.thumbnail {
  position: absolute; /* key difference */
  width: 100px;
  height: 100px;
  flex-shrink: 0; /* keep width fixed, can overflow scroll */
  margin: 0;
  text-align: center;
  color: #fff;
}
.thumbnail img {
  width: 100px;
  height: 60px;
  object-fit: cover;
}

/* The red playhead line at the top (z-index so it's on top of thumbnails) */
.playhead {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  background: red;
  pointer-events: none;
  z-index: 10;
}

.time-display {
  /* Force a fixed width so content doesn’t jump around */
  display: inline-block;
  width: 15ch;       /* or px/em — “ch” is based on the width of '0' */
  text-align: right;
  white-space: nowrap; /* keep it on one line */

  /* (Optional) use a monospaced font for consistent digit widths */
  font-family: monospace;
  flex-shrink: 0;

}

.gaze-dot {
  position: absolute;
  box-sizing: border-box;
  width: 20px;
  height: 20px;
  border-radius: 50%;       /* make it circular */
  border: 2px solid #ff0000;   /* 4px “weight” */
  background: transparent;  /* ring style */
  pointer-events: none;     /* ignore pointer/mouse */
}

.gaze-trace {
  position: absolute;
  top: 0;
  left: 0;
  pointer-events: none; /* so it doesn't block clicks */
  /* We’ll dynamically set canvas size in JS to match previewWidth x previewHeight */
}

/* The chart below the timeline. */
.chart-inner {
  display: block;
  height: 130px;
  width: 80vw;      /* 80% of the viewport width */
  max-width: 1250px;  /* but never exceed 1200px, for very wide screens */
  margin-left: -50px; /* example negative margin to shift left */
}

</style>

