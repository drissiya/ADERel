import glob
import os
from collections import defaultdict

class Corpora(object):

    def __init__(self, folder1, track_num):
        extensions = {1: '*.xml', 2: '*.ann'}
        file_ext = extensions[track_num]
        self.track = track_num
        self.folder1 = folder1
		
        files1 = set([os.path.basename(f) for f in glob.glob(
            os.path.join(folder1, file_ext))])

        self.docs = []
        for file in files1:
            g = RecordTrack2(os.path.join(self.folder1, file))
            self.docs.append(g)
			
class ClinicalConcept(object):
    """Named Entity Tag class."""

    def __init__(self, tid, start, end, ttype, text=''):
        """Init."""
        self.tid = str(tid).strip()
        self.start = int(start)
        self.end = int(end)
        self.text = str(text).strip()
        self.ttype = str(ttype).strip()

    def span_matches(self, other, mode='strict'):
        """Return whether the current tag overlaps with the one provided."""
        assert mode in ('strict', 'lenient')
        if mode == 'strict':
            if self.start == other.start and self.end == other.end:
                return True
        else:   # lenient
            if (self.end > other.start and self.start < other.end) or \
               (self.start < other.end and other.start < self.end):
                return True
        return False

    def equals(self, other, mode='strict'):
        """Return whether the current tag is equal to the one provided."""
        assert mode in ('strict', 'lenient')
        return other.ttype == self.ttype and self.span_matches(other, mode)

    def __str__(self):
        """String representation."""
        return '{}\t{}\t({}:{})'.format(self.ttype, self.text, self.start, self.end)


class Relation(object):
    """Relation class."""

    def __init__(self, rid, arg1, arg2, rtype):
        """Init."""
        assert isinstance(arg1, ClinicalConcept)
        assert isinstance(arg2, ClinicalConcept)
        self.rid = str(rid).strip()
        self.arg1 = arg1
        self.arg2 = arg2
        self.rtype = str(rtype).strip()

    def equals(self, other, mode='strict'):
        """Return whether the current tag is equal to the one provided."""
        assert mode in ('strict', 'lenient')
        if self.arg1.equals(other.arg1, mode) and \
                self.arg2.equals(other.arg2, mode) and \
                self.rtype == other.rtype:
            return True
        return False

    def __str__(self):
        """String representation."""
        return '{} ({}->{})'.format(self.rtype, self.arg1.ttype,
                                    self.arg2.ttype)

			
class RecordTrack2(object):
    """Record for Track 2 class."""

    def __init__(self, file_path):
        """Initialize."""
        self.path = os.path.abspath(file_path)
        self.basename = os.path.basename(self.path)
        self.annotations = self._get_annotations()
        # self.text = self._get_text()

    @property
    def tags(self):
        return self.annotations['tags']

    @property
    def relations(self):
        return self.annotations['relations']

    def _get_annotations(self):
        """Return a dictionary with all the annotations in the .ann file."""
        annotations = defaultdict(dict)
        with open(self.path) as annotation_file:
            lines = annotation_file.readlines()
            for line_num, line in enumerate(lines):
                if line.strip().startswith('T'):
                    try:
                        tag_id, tag_m, tag_text = line.strip().split('\t')
                    except ValueError:
                        print(self.path, line)
                    if len(tag_m.split(' ')) == 3:
                        tag_type, tag_start, tag_end = tag_m.split(' ')
                    elif len(tag_m.split(' ')) == 4:
                        tag_type, tag_start, _, tag_end = tag_m.split(' ')
                    elif len(tag_m.split(' ')) == 5:
                        tag_type, tag_start, _, _, tag_end = tag_m.split(' ')
                    else:
                        print(self.path)
                        print(line)
                    tag_start, tag_end = int(tag_start), int(tag_end)
                    annotations['tags'][tag_id] = ClinicalConcept(tag_id,
                                                                  tag_start,
                                                                  tag_end,
                                                                  tag_type,
                                                                  tag_text)
            for line_num, line in enumerate(lines):
                if line.strip().startswith('R'):
                    rel_id, rel_m = line.strip().split('\t')
                    rel_type, rel_arg1, rel_arg2 = rel_m.split(' ')
                    rel_arg1 = rel_arg1.split(':')[1]
                    rel_arg2 = rel_arg2.split(':')[1]
                    arg1 = annotations['tags'][rel_arg1]
                    arg2 = annotations['tags'][rel_arg2]
                    annotations['relations'][rel_id] = Relation(rel_id, arg1,
                                                                arg2, rel_type)
        return annotations

    def _get_text(self):
        """Return the text in the corresponding txt file."""
        path = self.path.replace('.ann', '.txt')
        with open(path) as text_file:
            text = text_file.read()
        return text

    def search_by_id(self, key):
        """Search by id among both tags and relations."""
        try:
            return self.annotations['tags'][key]
        except KeyError():
            try:
                return self.annotations['relations'][key]
            except KeyError():
                return None
