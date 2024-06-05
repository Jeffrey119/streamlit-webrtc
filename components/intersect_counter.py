import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import pandas as pd
import numpy as np
import copy
from typing import List
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from supervision.annotators.utils import ColorLookup, Trace, resolve_color
from ultralytics import YOLO, solutions
from components import session_result
from collections import defaultdict
from ultralytics.utils.plotting import Annotator, colors

from shapely.geometry import LineString, Point, Polygon

COLORS = np.random.uniform( 0, 255, size=(1024, 3) )

class point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __mul__(self, other: float):
        self.x *= other
        self.y *= other
        return self

    __rmul__ = __mul__

    def __add__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def __iter__(self):
        for i in [self.x, self.y]:
            yield i

    def __len__(self):
        return 2
    
    def __getitem__(self, idx):
        return (self.x, self.y)[idx]


# https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
def ccw(A, B, C):
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)


class IntersectCounter(solutions.ObjectCounter):
    def __init__(self, vertices: List[point], id, classes_names, scale: float = 1.0, ):
        self.vertices = [v * scale for v in vertices]
        # Init Object Counter
        super().__init__(
            view_img=False,
            reg_pts=self.vertices,
            classes_names=classes_names,
            draw_tracks=True,
            line_thickness=2,
        )
        self.id = int( id )
        self.track_in_frame = {}

    @classmethod
    def init_centroid(cls, centroid, x_offset, y_offset, id, class_names, scale=1.0):
        p1, p2 = copy.deepcopy( centroid ), copy.deepcopy( centroid )
        vertices = [p1 + point( x_offset, y_offset ), p2 + point( -x_offset, -y_offset )]
        return cls( vertices, id, class_names, scale )

    @DeprecationWarning
    def check_intersect(self, path_vertices: List[point], object):
        A, B, C, D = self.vertices[0], self.vertices[1], path_vertices[0], path_vertices[1]
        if ccw( A, C, D ) != ccw( B, C, D ) and ccw( A, B, C ) != ccw( A, B, D ):
            self.count += 1
            self.counted_objects.append( object )
        return self.count

    def reset(self):
        # Object counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = {}

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_in_frame = {}

    def update_counters(self, frame, tracks, frame_no, override_counter = None):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        if override_counter is not None:
            counters = override_counter # cannot access st.session state in webrtc

        self.im0 = frame
        self.extract_and_process_tracks(tracks, frame_no)
        if self.view_img:
            self.display_frames()
        return self.im0

    def counted_objects (self):
        return self.count_ids
    
    # override original method by ultralytics
    def extract_and_process_tracks(self, tracks, frame_no):
        """Extracts and processes tracks for object counting in a video stream."""

        # Annotator Init and region drawing
        self.annotator = Annotator(self.im0, self.tf, self.names)

        # Draw region or line
        self.annotator.draw_region(reg_pts=self.reg_pts, color=self.region_color, thickness=self.region_thickness)

        # Annotate Frame Number
        font = cv2.FONT_HERSHEY_DUPLEX
        color = (255, 120, 0) # red
        fontsize = 10
        text = F"Frame:{frame_no}"
        position = (50, 50)
        self.annotator.text(position, text, color)

        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            # Extract tracks
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # Draw bounding box
                self.annotator.box_label(box, label=f"{self.names[cls]}#{track_id}", color=colors(int(track_id), True))

                # Store class info
                if self.names[cls] not in self.class_wise_count:
                    self.class_wise_count[self.names[cls]] = {"IN": [], "OUT": []}

                # Store Frame number
                if track_id not in self.track_in_frame:
                    self.track_in_frame[track_id] = [frame_no]
                else:
                    self.track_in_frame[track_id].append(frame_no)

                # Draw Tracks
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                     track_line.pop(0)

                # Draw track trails
                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line,
                        color=self.track_color if self.track_color else colors(int(track_id), True),
                        track_thickness=self.track_thickness,
                    )

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                # Count objects in any polygon
                if len(self.reg_pts) >= 3:
                    is_inside = self.counting_region.contains(Point(track_line[-1]))

                    if prev_position is not None and is_inside and track_id not in self.count_ids:
                        self.count_ids.append(track_id)

                        if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                            self.in_counts += 1
                            self.class_wise_count[self.names[cls]]["IN"].append(track_id)
                        else:
                            self.out_counts += 1
                            self.class_wise_count[self.names[cls]]["OUT"].append(track_id)

                # Count objects using line
                elif len(self.reg_pts) == 2:
                    if prev_position is not None and track_id not in self.count_ids:
                        distance = Point(track_line[-1]).distance(self.counting_region)
                        if distance < self.line_dist_thresh and track_id not in self.count_ids:
                            self.count_ids.append(track_id)

                            if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                                self.in_counts += 1
                                self.class_wise_count[self.names[cls]]["IN"].append(track_id)
                            else:
                                self.out_counts += 1
                                self.class_wise_count[self.names[cls]]["OUT"].append(track_id)

        labels_dict = {}

        for key, value in self.class_wise_count.items():
            if len(value["IN"]) != 0 or len(value["OUT"]) != 0:
                if not self.view_in_counts and not self.view_out_counts:
                    continue
                elif not self.view_in_counts:
                    labels_dict[str.capitalize(key)] = f"OUT {len(value['OUT'])}"
                elif not self.view_out_counts:
                    labels_dict[str.capitalize(key)] = f"IN {len(value['IN'])}"
                else:
                    labels_dict[str.capitalize(key)] = f"IN {len(value['IN'])} OUT {len(value['OUT'])}"

        if labels_dict:
            self.annotator.display_analytics(self.im0, labels_dict, self.count_txt_color, self.count_bg_color, 10)




class st_IntersectCounter:
    def __init__(self, image, id, width, height, screen_width=1200):
        self.user_num_input = None
        self.user_cat_input = None
        self.to_filter_columns = None
        self.modification_container = None
        self.left = None
        self.right = None
        self.tab_id = id
        if 'counters_tables' not in st.session_state:
            st.session_state.counters_tables = {}
            self.counters_table = None
        else:
            self.counters_table = st.session_state.counters_tables.get(self.tab_id,None)
        if 'counters' not in st.session_state:
            st.session_state.counters = {}
        self.display_scale = width / screen_width *0.9
        self.counters_df_display = None
        self.counters_num = 0
        self.wrapper = st.expander( "**Setup Counter**" , expanded=True)
        self.option = 'Empty'
        self.counter_result_display = None
        screen_height = height // self.display_scale
        with self.wrapper:
            st.caption( "Draw lines on the below picture to set up counting function" )
            canvas_result = st_canvas(
                width=screen_width,
                height=screen_height,
                background_image=Image.fromarray( cv2.cvtColor( image, cv2.COLOR_BGR2RGB ) ),
                stroke_width=1,
                drawing_mode='line', key="canvas"+str(self.tab_id)
            )

            if canvas_result.json_data is not None:
                if len( canvas_result.json_data['objects'] ) > 0:
                    self.canvas_result = canvas_result.json_data['objects']
                    self.counters_num = len( self.canvas_result )
                    self.format_counters_display()
                    all( self.generate_counters() )
                    st.markdown( '**Screenline Counters**' )    
            if st.session_state.counters_tables.get(self.tab_id) is not None:
                self.counters_df_display = st.dataframe(
                    st.session_state.counters_tables.get(self.tab_id).style.format( precision=1 ), use_container_width=True, )
                        #st.markdown( '**Screenline Counters Result**' )

    def generate_counters(self):
        # if not st.session_state.counted:
        #     st.session_state.counters = []
        existed_counter = st.session_state.counters.get(self.tab_id, None)
        if existed_counter is None:
            st.session_state.counters[self.tab_id] = []
            existed_counter=st.session_state.counters.get(self.tab_id)
        for ind in range( len( existed_counter ), self.counters_num ):
            centroid = point( self.counters_table['left'][ind], self.counters_table['top'][ind] )
            xoffset = self.counters_table['x1'][ind]
            yoffset = self.counters_table['y1'][ind]
            counter = IntersectCounter.init_centroid( centroid, xoffset, yoffset, ind, st.session_state.class_names, self.display_scale )
            existed_counter.append( counter )
            self.sync_session_state(existed_counter)
            yield counter

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a UI on top of a dataframe to let viewers filter columns
        https://github.com/tylerjrichards/st-filter-dataframe

        Args:
            df (pd.DataFrame): Original dataframe

        Returns:
            pd.DataFrame: Filtered dataframe
        """

        df = df.copy()

        if 'to_filter_columns' not in st.session_state:
            st.session_state.to_filter_columns = None
        if 'user_cat_input' not in st.session_state:
            st.session_state.user_cat_input = None

        # Try to convert datetimes into a standard format (datetime, no timezone)
        for col in df.columns:
            if is_object_dtype( df[col] ):
                try:
                    df[col] = pd.to_datetime( df[col] )
                except Exception:
                    pass

            if is_datetime64_any_dtype( df[col] ):
                df[col] = df[col].dt.tz_localize( None )

        if self.modification_container is None:
            self.modification_container = st.container()

        with self.modification_container:
            if self.to_filter_columns is None:
                self.to_filter_columns = st.multiselect( "Filter dataframe on", df.columns )
            for column in self.to_filter_columns:
                if self.left is None:
                    self.left, self.right = st.columns( (1, 20) )
                # Treat columns with < 10 unique values as categorical
                if is_categorical_dtype( df[column] ) or df[column].nunique() < 10:
                    if self.user_cat_input is None:
                        self.user_cat_input = self.right.multiselect(
                            f"Values for {column}",
                            df[column].unique(),
                            default=list( df[column].unique() ),
                        )
                    df = df[df[column].isin( self.user_cat_input )]
                elif is_numeric_dtype( df[column] ):
                    _min = float( df[column].min() )
                    _max = float( df[column].max() )
                    step = (_max - _min) / 100
                    if self.user_num_input is None:
                        self.user_num_input = user_num_input = self.right.slider(
                            f"Values for {column}",
                            min_value=_min,
                            max_value=_max,
                            value=(_min, _max),
                            step=step,
                            key='user_num_input'
                        )
                    df = df[df[column].between( *self.user_num_input )]
                elif is_datetime64_any_dtype( df[column] ):
                    user_date_input = self.right.date_input(
                        f"Values for {column}",
                        value=(
                            df[column].min(),
                            df[column].max(),
                        ),
                    )
                    if len( user_date_input ) == 2:
                        user_date_input = tuple( map( pd.to_datetime, user_date_input ) )
                        start_date, end_date = user_date_input
                        df = df.loc[df[column].between( start_date, end_date )]
                else:
                    user_text_input = self.right.text_input(
                        f"Substring or regex in {column}",
                    )
                    if user_text_input:
                        df = df[df[column].astype( str ).str.contains( user_text_input )]

        return df

    def show_counter_results(self):
        if len( st.session_state.counters.get(self.tab_id, None) ) > 0:
            result_df = self.filter_dataframe(session_result.result_to_df(
                [d.counted_objects for d in st.session_state.counters.get(self.tab_id, None)], counter_column=True) )
            result_df = result_df.style.background_gradient( axis=0, gmap=result_df['counter'], cmap='BuPu' )
            if self.counter_result_display is not None:
                self.counter_result_display.dataframe( result_df, use_container_width=True )
            else:
                self.counter_result_display = st.dataframe( result_df, use_container_width=True )

            self.counters_df_display.dataframe(
                self.format_counters_display().style.format( precision=1 ) )

    def format_counters_display(self):
        self.counters_table = pd.json_normalize( self.canvas_result )
        show_columns = ['type', 'left', 'top', 'x1', 'x2', 'y1', 'y2', 'width', 'height']
        if self.counters_table is not None:
            self.counters_table = self.counters_table[show_columns]
            # self.counters_table.style.format(precision=1)
        else:
            return None
        if len( st.session_state.counters.get(self.tab_id,[]) ) == len( self.counters_table ) and len( st.session_state.counters.get(self.tab_id,[]) ) > 0:
            self.counters_table['in_counts'] = [r.in_counts for r in st.session_state.counters[self.tab_id]]
            self.counters_table['out_counts'] = [r.out_counts for r in st.session_state.counters[self.tab_id]]
        self.sync_session_state()
        return self.counters_table

    def sync_session_state(self, existed_counter=None):
        if existed_counter:
            st.session_state.counters[self.tab_id] = existed_counter
        st.session_state.counters_tables[self.tab_id] = self.counters_table

