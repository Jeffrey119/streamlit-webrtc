import config
from components import intersect_counter as ic, variables_panel as vp, progress_bar as pb, session_result

import platform
import logging
import queue
import ffmpeg
import subprocess
import tempfile
import uuid
import av
import cv2
import numpy as np
from io import BytesIO
import streamlit as st
from aiortc.contrib.media import MediaPlayer
import sys
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
from streamlit_drawable_canvas import st_canvas
from streamlit.runtime.scriptrunner.script_run_context import add_script_run_ctx, get_script_run_ctx
from PIL import Image
from st_tabs import TabBar
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from super_image import EdsrModel, ImageLoader
import torch.nn.functional as F
from cv2 import dnn_superres
import torch

sys.setrecursionlimit( 4000 )

from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    webrtc_streamer,
)

st.set_page_config( layout="wide" )
logger = logging.getLogger( __name__ )

# func to save BytesIO on a drive
def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())

def run_tracker_in_thread(cap, model, counter, tab_id, conf, iou, CLASS_ID, progress, ctx):
    """
    Runs a video file or webcam stream concurrently with the YOLOv8 model using threading.

    This function captures video frames from a given file or camera source and utilizes the YOLOv8 model for object
    tracking. The function runs in its own thread for concurrent processing.

    Args:
        filename (str): The path to the video file or the identifier for the webcam/external camera source.
        model (obj): The YOLOv8 model object.
        file_index (int): An index to uniquely identify the file being processed, used for display purposes.

    Note:
        Press 'q' to quit the video display window.
    """

    add_script_run_ctx(threading.currentThread(), ctx)

    width, height = int( cap.get( cv2.CAP_PROP_FRAME_WIDTH ) ), int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
    fps = cap.get( cv2.CAP_PROP_FPS )
    output_path = os.path.join( config.HERE, f"storage\\{str( uuid.uuid4() )}.mp4" )

    # encode cv2 output into h264
    # https://stackoverflow.com/questions/30509573/writing-an-mp4-video-using-python-opencv
    args = (ffmpeg
            .input( 'pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format( width, height ) )
            .output( output_path, pix_fmt='yuv420p', vcodec='libx264', r=fps, crf=12 )
            .overwrite_output()
            .get_args()
            )

    # check if deployed on cloud or local host
    ffmpeg_source = config.FFMPEG_PATH if platform.processor() else 'ffmpeg'
    process = subprocess.Popen( [ffmpeg_source] + args, stdin=subprocess.PIPE )

    us_model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)

    while True:
        ret, frame = cap.read()  # Read the video frames

        # Exit the loop if no more frames in either video
        if not ret:
            break
        
        # inputs = ImageLoader.load_image(Image.fromarray(frame))
        # lr = frame[::].astype(np.float32).transpose([2, 0, 1]) / 255.0
        # inputs = torch.as_tensor(np.array([lr]))
        # pred = us_model(inputs)
# # 
        # # Up-scale image
        # pred = pred.data.cpu().numpy()
        # pred = pred[0].transpose((1, 2, 0)) * 255.0
        # pred = pred.astype(np.uint8)
        # pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        # frame = pred

        # Apply your low-light enhancement algorithm (example with simple histogram equalization)
        # frame = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # Track objects in frames if available
        results = model.track(frame, persist=True, conf=conf, iou=iou, classes=CLASS_ID)
        annotated_frame = results[0].plot()
        if len(counter):
            for icounter in counter:
                annotated_frame = icounter.start_counting(annotated_frame, results)
        
        annotated_frame = cv2.resize(annotated_frame, (width, height)) 
        process.stdin.write( cv2.cvtColor( annotated_frame, cv2.COLOR_BGR2RGB ).astype( np.uint8 ).tobytes() )
        # st.session_state['result_list'].extend( results )

        progress.progress(1)
#

    # Release video sources  
    process.stdin.close()
    process.wait()
    process.kill()
    cap.release()

    # TODO: skip writing the analysis result into a temp file and read into memory
    with open( output_path, "rb" ) as fh:
        buf = BytesIO( fh.read() )
    os.remove( output_path )

    return buf

def video_object_detection(variables):
    """
    Static detection method on the uploaded video
    https://github.com/yeha98555/object-detection-web-app
    """
    # test video for detection
    # https://www.pexels.com/video/aerial-footage-of-vehicular-traffic-of-a-busy-street-intersection-at-night-3048225/

    weight, confidence_threshold, track_age, track_hits, iou_thres = variables.get_var()

    if 'result_list' not in st.session_state:
        st.session_state.result_list = []
    if 'videos' not in st.session_state:
        st.session_state.videos = []
    if 'files' not in st.session_state:
        st.session_state.files = []

    files = st.file_uploader( 'Upload Videos', type=['avi', 'mp4', 'mov'] , accept_multiple_files=True)
    if len(files):
        if files != st.session_state.files:
            st.session_state.files = files
            st.session_state.videos = []
            st.session_state.counters = {}
            st.session_state.counters_tables = {}
            st.session_state.counted = False
            st.session_state.result_list = {}

        icounters = {}
        caps = {}

        # temp dir for saving the video to be processed by opencv
        if not os.path.exists( os.path.join( config.HERE, 'storage' ) ):
            os.makedirs( os.path.join( config.HERE, 'storage' ) )
        # save the uploaded file to a temporary location
        for i, f in enumerate(files):
            temp_file_to_save = f'./storage/temp_file_{i}.mp4'
            write_bytesio_to_file(temp_file_to_save, f)
            caps[i] = cv2.VideoCapture(temp_file_to_save)

        # class_ids of interest - car, motorcycle, bus and truck
        CLASS_ID = [2,3, 5, 7]
        # init object detector and tracker
        model = YOLO(config.STYLES[weight])
        model.classes=CLASS_ID
        # dict maping class_id to class_name
        st.session_state.class_names = model.names

        tabs = TabBar(tabs=[f.name for f in files], default=0)
 

        width, height = int( caps[tabs].get( cv2.CAP_PROP_FRAME_WIDTH ) ), int( caps[tabs].get( cv2.CAP_PROP_FRAME_HEIGHT ) )
        success, image = caps[tabs].read()

        # setup expander for user to draw line counters
        icounters[id] = ic.st_IntersectCounter( image, tabs, width, height )

        # size limited by streamlit cloud service (superseded)
        # if max( width, height ) > 1920 or min( width, height ) > 1080:
        #     st.warning( f"File resolution [{width}x{height}] exceeded limit [1920x1080], "
        #                 f"please consider scale down the video", icon="⚠️" )
        # else:
        detect = st.button( 'Detect' )
        if detect:
            # reset all counters
            for t, cs in st.session_state.counters.items():
                for c in cs:
                    c.reset()   

            progress_bars = {}
            iterables = []

            for i, cap in enumerate(caps):
                # Create the tracker threads
                progress_bars[i] = pb.Progress_bar(i, caps[i].get( cv2.CAP_PROP_FRAME_COUNT ))
                counter = st.session_state.counters.get(i, [])
                iterables.append([caps[i], model, counter, i, confidence_threshold, iou_thres, CLASS_ID, progress_bars[i]])
                
            
            # Use ThreadPoolExecutor to manage concurrent execution
            with ThreadPoolExecutor(max_workers=8) as executor:
                ctx = get_script_run_ctx()
                # Submit tasks for execution
                futures = [executor.submit(run_tracker_in_thread, *x, ctx) for x in iterables]

                # Collect results from completed tasks
                results = [future.result() for future in as_completed(futures)]

            
            st.session_state.counted = True
            st.session_state.videos = results

            for k,p in progress_bars.items():
                p.terminate()

        if len(st.session_state.videos):
            st.video( st.session_state.videos[tabs] )

        # Dumping analysis result into table
        if st.session_state.counted:
            if st.checkbox( "Show all detection results" ):
                if len( st.session_state.result_list ) > 0:
                    result_df = session_result.result_to_df( st.session_state.result_list )
                    st.dataframe( result_df, use_container_width=True )
            # icounter.show_counter_results()


@st.cache
def load_model(weight, conf):
    detector = detection_helpers.Detector( conf )
    detector.load_model( 'weights/' + config.STYLES[weight], trace=False )
    return detector


def live_object_detection(variables):
    """
    #This component was originated from https://github.com/whitphx/streamlit-webrtc
    """
    weight, confidence_threshold, track_age, track_hits, iou_thres = variables.get_var()

    # init frame counter, object detector, tracker and passing object counter
    frame_num = fc.FrameCounter()
    detector = load_model( weight, confidence_threshold )
    deepsort_tracker = bridge_wrapper.YOLOv7_DeepSORT(
        reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector,
        max_iou_distance=iou_thres, max_age=track_age, n_init=track_hits )

    if 'counters' not in st.session_state:
        st.session_state.counters = []
    icounter = st.session_state.counters

    # Dump queue for real time detection result
    result_queue = (queue.Queue())
    frame_queue = (queue.Queue( maxsize=1 ))

    # reading each frame of live stream and passing to backend processing
    def frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        frame = frame.to_ndarray( format="bgr24" )

        # Detect, track and counter the intersect of objects here
        image, result = deepsort_tracker.track_video_stream( frame, frame_num( 1 ) )
        if icounter is not None:
            if len( icounter ) > 0:
                image = st_icounter.update_counters( deepsort_tracker.tracker.tracks, image, icounter )

        # NOTE: This `recv` method is called in another thread,
        # so it must be thread-safe.
        result_queue.put( result )
        if not frame_queue.full():
            frame_queue.put( image )

        return av.VideoFrame.from_ndarray( image, format="bgr24" )

    # public-stun-list.txt
    # https://gist.github.com/mondain/b0ec1cf5f60ae726202e
    servers = [{"url": "stun:stun.l.google.com:19302"}]
    if 'URL' in st.secrets:
        servers.append( {"urls": st.secrets['URL'],
                         "username": st.secrets['USERNAME'],
                         "credential": st.secrets['CREDENTIAL'], } )
    RTC_CONFIGURATION = RTCConfiguration( {"iceServers": servers} )

    # RTSP video source
    if st.checkbox( 'RTSP' ):
        url = st.text_input( 'RTSP URL' )
        media_file_info = {
            "url": url,
            "type": "video",}

        def create_player():
            return MediaPlayer( media_file_info["url"] )

        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.RECVONLY,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={
                "video": media_file_info["type"] == "video",
                "audio": media_file_info["type"] == "audio",
            },
            player_factory=create_player,
            video_frame_callback=frame_callback,
        )

    else:
        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,  # when deploy on remote host need stun server for camera connection
            video_frame_callback=frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    # capture image for the counter setup container
    if webrtc_ctx.state.playing:
        image = frame_queue.get()
        st_icounter = ic.st_IntersectCounter( image, image.shape[1], image.shape[0] )
        icounter = st.session_state.counters
        if len( st.session_state.counters ) > 0:
            st.session_state.counted = True
        labels_placeholder = st.empty()

        # NOTE: The video transformation with object detection and
        # this loop displaying the result labels are running
        # in different threads asynchronously.
        # Then the rendered video frames and the labels displayed here
        # are not strictly synchronized.
        while True:
            try:
                result = result_queue.get( timeout=1.0 )
                labels_placeholder.dataframe( session_result.result_to_df( result ), use_container_width=True )
            except queue.Empty:
                result = None
            if st_icounter is not None:
                st_icounter.show_counter_results()

    else:
        st.session_state.counters = []
        st.session_state.counters_table = []
        st.session_state.counted = False
        st.session_state.result_list = []


def main():
    st.header( "Object Detecting and Tracking demo" )

    pages = {
        "Upload Video for detection": video_object_detection,
        "Real time object detection (sendrecv)": live_object_detection,
    }
    page_titles = pages.keys()

    my_sidebar = st.sidebar
    page_title = my_sidebar.selectbox(
        "Choose the app mode",
        page_titles,
    )

    with my_sidebar:
        variables = vp.st_VariablesPanel()
    st.subheader( page_title )

    page_func = pages[page_title]
    page_func( variables )

    # logger.debug( "=== Alive threads ===" )
    # for thread in threading.enumerate():
    #    if thread.is_alive():
    #        logger.debug( f"  {thread.name} ({thread.ident})" )


if __name__ == "__main__":
    import os

    DEBUG = config.DEBUG

    if DEBUG:
        logging.basicConfig(
            format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
                   "%(message)s",
            force=True,
        )

        logger.setLevel( level=logging.DEBUG if DEBUG else logging.INFO )

        st_webrtc_logger = logging.getLogger( "streamlit_webrtc" )
        st_webrtc_logger.setLevel( logging.DEBUG )

        fsevents_logger = logging.getLogger( "fsevents" )
        fsevents_logger.setLevel( logging.WARNING )

    main()
