# Datasheet for dataset "Fluorescent Neuronal Cells v2"

Questions from the [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) paper, v7. Template from [GitHub template](https://github.com/fau-masters-collected-works-cgarbin/datasheet-for-dataset-template).

Jump to section:

- [Motivation](#motivation)
- [Composition](#composition)
- [Collection process](#collection-process)
- [Preprocessing/cleaning/labeling](#preprocessingcleaninglabeling)
- [Uses](#uses)
- [Distribution](#distribution)
- [Maintenance](#maintenance)

---
## Motivation

_The questions in this section are primarily intended to encourage dataset creators
to clearly articulate their reasons for creating the dataset and to promote transparency
about funding interests._

### For what purpose was the dataset created? 

_Was there a specific task in mind? Was there a specific gap that needed to be filled?
Please provide a description._

    Many life science studies rely on human operators to analyze fluorescence microscopy images and attempt to recognize and count the relevant biological structures. The FNC v2 data was created to support applications that use this imaging technique, specifically targeting neuronal cells. It provides annotated data in various formats to facilitate the development of automated approaches based on supervised learning.

    Potential anticipated extensions might include:
     - transfer learning (fluorescence microscopy images involving other biological structures)
     - transfer learning (neuronal cells pictures acquired with a different imaging techinque)
     - methodological approaches to semantic segmentation, object detection and counting
     - methodological studies about how to make efficient use of data, e.g. Transfer Learning, Weakly Supervised Learning, Self Supervised Learning, Few-Shot Learning, and more

### Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?

<pre>
The dataset was curated through a collaborative effort between researchers from the Italian National Institute of Nuclear Physics (INFN) and the University of Bologna, Department of Biomedical and Neuromotor Sciences and Department of Physics and Astronomy. 
For more details and authors list, please visit the <a href="https://amsacta.unibo.it/id/eprint/7347 ">official repository</a>.
</pre>

### Who funded the creation of the dataset? 

_If there is an associated grant, please provide the name of the grantor and the grant
name and number._

    The collection of the original fluorescence images was supported by funding from the University of Bologna (RFO 2018) and the European Space Agency (Research agreement collaboration 4000123556).
    The collection of the corresponding ground truth annotations was conducted by researchers of the National Institute for Nuclear Physics and the University of Bologna.

### Any other comments?

    None

--- 
## Composition

_Most of these questions are intended to provide dataset consumers with the
information they need to make informed decisions about using the dataset for
specific tasks. The answers to some of these questions reveal information
about compliance with the EU’s General Data Protection Regulation (GDPR) or
comparable regulations in other jurisdictions._

### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?

    The fundamental data units are pictures of rodent brain slices. These are acquired applying different light wavelenght filters to the fluorescence microscope before taking the shot. 

    Our data features three different filters, namely *green*, *red* and *yellow*, which gives the names to the three datasets.
    Notice that the same slice may be pictured multiple times with different filters. When that is the case, the original files (*raw_data folder*) share the same filename except for the last letter that represent the filter color. 
    For example, Mar22bS1C2R1_VLPAGl_200x_g.TIF and Mar22bS1C2R1_VLPAGl_200x_y.TIF are pictures of the same slice (Mar22bS1C2R1_VLPAGl_200x), obtained with a different filter (`_g` stands for green and `_y` stands for yellow).

    Images with this characteristic are said "double marked". In order to recover them, please refer to `metadata_v2.xlsx` map.

_Are there multiple types of instances (e.g., movies, users, and ratings; people and
interactions between them; nodes and edges)? Please provide a description._

    Inside each image, the marked biological structures may vary depending on the specific wavelength we observe. This can be:
     - cell nucleus --> green
     - cell cytoplasm --> red
     - cell cytoplasm --> yellow

### How many instances are there in total (of each type, if appropriate)?

    The Fluorescent Neuronal Cells data collection comprises 1874 fluorescence microscopy pictures, divided into three datasets of 691 (green), 546 (red) and 637 (yellow) images. Among these, we provide 750 ground-truth annotations (283 for green and yellow, 184 for red). The remaining 1124 images are unlabelled.

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?

_If the dataset is a sample, then what is the larger set? Is the sample representative
of the larger set (e.g., geographic coverage)? If so, please describe how this
representativeness was validated/verified. If it is not representative of the larger set,
please describe why not (e.g., to cover a more diverse range of instances, because
instances were withheld or unavailable)._

    The images are obtained by sectioning rodent brains into 35 μm thick tissue slices, with sampling conducted at regular intervals of 105 μm for mice (green and yellow) and of 210 μm for rats (red). This was done to avoid redundant data and ensure comprehensive coverage while maintaining manageable data size.

    Also, only some brain regions were observed depending on their relevance to the study of torpor mechanisms, namely Raphe Pallidus (RPa), Dorsomedial Hypothalamus (DMH), Lateral Hypotalamus (LH), and Ventrolateral Periaqueductal Gray (VLPAG).

    The resulting data are therefore considered representative of those regions under the experimental settings adopted. Please refer to the FNC v2 original paper for more details.

### What data does each instance consist of? 

_“Raw” data (e.g., unprocessed text or images) or features? In either case, please
provide a description._

    The suggested individual instances consist of *.png* images available under the `images` folder inside the `green`, `red` and `yellow` directories. 
    These files are uncompressed conversion of the original *.TIF* pictures, and keep all relevant metadata.
    
    For reproducibility, we share also the raw data coming from image acquisition (raw_data.zip) and data annotation stages (annotations.zip).

### Is there a label or target associated with each instance?

_If so, please provide a description._

    We provide ground-truth labels for 750 original images. These are available for each dataset under the `<data-split>/ground_truths/` folder, where `<data-split>` can be `trainval` or `test`.
    
    The association between image and corresponding ground-truth can be done based on the numbered filename. 
    
    Finally, notice that multiple annotation types and formats are provided inside the `ground_truths` directory:
        
        - masks/: binary masks; .png files; one per image
        - rle/: binary mask as RLE encoding; pickle files; one per image
        - Pascal VOC/: polygon, bounding box, dot annotation and count; xml files; one per image 
        - COCO/:  polygon, bounding box, dot annotation and count; json file; one data split 
        - VIA/:  polygon; json file; one data split 

### Is any information missing from individual instances?

_If so, please provide a description, explaining why this information is missing (e.g.,
because it was unavailable). This does not include intentionally removed information,
but might include, e.g., redacted text._

    For red images, some original filenames starting by `RT` (rats samples) present brain regions encoded as `1`/`2` or `a`/`b`. These all refer to Raphae Pallidus (RPa) with a dorsal and a ventral framing.

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?

_If so, please describe how these relationships are made explicit._

    The relations are due to the possible linkage to same brain slice described above.

### Are there recommended data splits (e.g., training, development/validation, testing)?

_If so, please provide a description of these splits, explaining the rationale behind them._

    Data split is pre-made and considered fixed for reproducibility reasons. This was done randomly, except for red images where sampling was balanced to replicate the rat/mice proportions in trainval and test splits. 

    The yellow data split replicates the one adopted in Morelli, R. et al., SciRep (2021), [Automating cell counting in fluorescent microscopy through deep learning with c-ResUnet.](https://www.nature.com/articles/s41598-021-01929-5) However, the results are not fully reproducible since the ground-truth masks were revised in v2.

### Are there any errors, sources of noise, or redundancies in the dataset?

_If so, please provide a description._


<pre>
The principal source of noise is due to the intrinsic subjectivity of the recognition task. This produces noise in labeled objects since similar objects may be annotated differently by different annotators, or by the same annotator over time.

Another potential source of noise is the blurred appearance caused by the specific acquisition technique (<i>epifluoresce</i>). 

No redundancy effects are known at the time of this writing.
</pre>

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?

_If it links to or relies on external resources, a) are there guarantees that they will
exist, and remain constant, over time; b) are there official archival versions of the
complete dataset (i.e., including the external resources as they existed at the time the
dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with
any of the external resources that might apply to a future user? Please provide descriptions
of all external resources and any restrictions associated with them, as well as links or other
access points, as appropriate._

    The Fluorescent Neuronal Cells v2 data collection is self-contained.


### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)?

_If so, please provide a description._

    NO

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

_If so, please describe why._

    NO

### Does the dataset relate to people? 

_If not, you may skip the remaining questions in this section._

    NO

### Does the dataset identify any subpopulations (e.g., by age, gender)?

_If so, please describe how these subpopulations are identified and provide a description of
their respective distributions within the dataset._

    NOT APPLICABLE

### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?

_If so, please describe how._

    NOT APPLICABLE

### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?

_If so, please provide a description._

    NOT APPLICABLE

### Any other comments?

    NONE

---
## Collection process

_\[T\]he answers to questions here may provide information that allow others to
reconstruct the dataset without access to it._

### How was the data associated with each instance acquired?

_Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g.,
survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags,
model-based guesses for age or language)? If data was reported by subjects or indirectly
inferred/derived from other data, was the data validated/verified? If so, please describe how._

    A total of 68 rodents were subjected to controlled experimental conditions to study torpor and thermoregulatory mechanisms. 
    At the end of the experimental session, the animals were deeply anaesthetized and transcardially perfused with 4% formaldehyde.
    This process allowed for the tagging of several neuronal substructures located within the nucleus or cytoplasm of the neurons.

    Rodents brains were then sectioned, and the resulting slices were finally stained for distinct markers following a standard immunofluorescence protocol.

### What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?

    The imaging technique adopted is called fluorescence microscopy. In our case, the experimental equipment was composed by a fluorescence microscope endowed with a high resolution camera. 
    Specifically, a Nikon Eclipse 80i microscope, equipped with a Nikon Digital Sight DS-Vi1 color camera, at a magnification of 200x (green and yellow).
    For red images, the same equipment was alterned with an ausJENA JENAVAL microscope, equipped with a Nikon Coolpix E4500 color camera, at a magnification of 250x. 
    Please refer to the FNC v2 paper for more details.

_How were these mechanisms or procedures validated?_

    Please refer to Hitrec, T., et al., SciRep (2019). [Neural control of fasting-induced torpor in mice](https://www.nature.com/articles/s41598-019-51841-2) for more information.

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?

    The images are obtained by sectioning rodent brains into 35 μm thick tissue slices, with sampling conducted at regular intervals of 105 μm for mice (green and yellow) and of 210 μm for rats (red). This was done to avoid redundant data and ensure comprehensive coverage while maintaining manageable data size.

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?

    Researchers from affiliated with the University of Bologna, regularly funded by research grants. Please see LiCENSE files for more details.

### Over what timeframe was the data collected?

_Does this timeframe match the creation timeframe of the data associated with the instances (e.g.
recent crawl of old news articles)? If not, please describe the timeframe in which the data
associated with the instances was created._

    The image data were acquired in a time span between 2016 and 2019.
    The ground-truth annotations were collected independently in 2023.

    Please refer to metadata of individual files for more details (`metadata/` folder inside each data split).

### Were any ethical review processes conducted (e.g., by an institutional review board)?

_If so, please provide a description of these review processes, including the outcomes, as well as
a link or other access point to any supporting documentation._

    All the experiments were conducted following approval by the ethical committee of National Health Authority. Mice and rats underwent experiments in different time and were subjected to different legislations for the ethical approvement of the experimental procedures: i) for rats, the experimental protocol was approved by the Ethical Committee for Animal Research of the University of Bologna and by the Italian Ministry of Health (decree: No.186/2013-B), and was performed in accordance with the European Union (2010/63/UE) and the Italian Ministry of Health (January 27, 1992, No. 116) Directives, under the supervision of the Central Veterinary Service of the University of Bologna and the National Health Authority; ii) for mice, the experimental protocol was approved by the National Health Authority (decree: No.141/2018 - PR/AEDB0.8.EXT.4), in accordance with the DL 26/2014 and the European Union Directive 2010/63/EU, and under the supervision of the Central Veterinary Service of the University of Bologna. All efforts were made to minimize the number of animals used and their pain and distress.


### Does the dataset relate to people?

_If not, you may skip the remainder of the questions in this section._

    NO

### Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?

    NOT APPLICABLE

### Were the individuals in question notified about the data collection?

_If so, please describe (or show with screenshots or other information) how notice was provided,
and provide a link or other access point to, or otherwise reproduce, the exact language of the
notification itself._

    NOT APPLICABLE

### Did the individuals in question consent to the collection and use of their data?

_If so, please describe (or show with screenshots or other information) how consent was
requested and provided, and provide a link or other access point to, or otherwise reproduce, the
exact language to which the individuals consented._

    NOT APPLICABLE

### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?

_If so, please provide a description, as well as a link or other access point to the mechanism
(if appropriate)._

    NOT APPLICABLE

### Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?

_If so, please provide a description of this analysis, including the outcomes, as well as a link
or other access point to any supporting documentation._

    NOT APPLICABLE

### Any other comments?

    The annotation process was conducted in compliance with a protocol specified prior to data annotation. For more details, please check the `Annotation protocol.pdf` file inside the `annotations.zip` archive.

---
## Preprocessing/cleaning/labeling

_The questions in this section are intended to provide dataset consumers with the information
they need to determine whether the “raw” data has been processed in ways that are compatible
with their chosen tasks. For example, text that has been converted into a “bag-of-words” is
not suitable for tasks involving word order._

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?

_If so, please provide a description. If not, you may skip the remainder of the questions in
this section._

    The original images were simply converted from *.TIF* to *.png* format to facilitate accessibility. This was done taking care of preserving EXIF metadata.
    For more details, please check `dataOps/convert_raw2png.py` script in the [GitHub repository](https://github.com/clissa/fluocells-scientific-data).


### Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?

_If so, please provide a link or other access point to the “raw” data._

    The raw data are maintained and shared as `raw_data.zip` and `annotations.zip` archives.

### Is the software used to preprocess/clean/label the instances available?

_If so, please provide a link or other access point._

    Yes, please check the [GitHub repository](https://github.com/clissa/fluocells-scientific-data).

### Any other comments?

    NONE

---
## Uses

_These questions are intended to encourage dataset creators to reflect on the tasks
for which the dataset should and should not be used. By explicitly highlighting these tasks,
dataset creators can help dataset consumers to make informed decisions, thereby avoiding
potential risks or harms._

### Has the dataset been used for any tasks already?

_If so, please provide a description._

    Yes, the dataset was used for object detection/segmentation in:
     - Morelli, R. et al., SciRep (2021), [Automating cell counting in fluorescent microscopy through deep learning with c-ResUnet.](https://www.nature.com/articles/s41598-021-01929-5)
     - Clissa, Luca (2022) [Supporting Scientific Research Through Machine and Deep Learning: Fluorescence Microscopy and Operational Intelligence Use Cases](http://amsdottorato.unibo.it/10016/), [Dissertation thesis]

### Is there a repository that links to any or all papers or systems that use the dataset?

_If so, please provide a link or other access point._

    NO

### What (other) tasks could the dataset be used for?

    Given the various annotation types provided, the FNC v2 data collection can be used for several learning problems, including semantic segmentation, object detection and counting, transfer learning, weakly/self- supervised learning, unsupervised learning, and more.

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

_For example, is there anything that a future user might need to know to avoid uses that
could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of
service issues) or other undesirable harms (e.g., financial harms, legal risks) If so, please
provide a description. Is there anything a future user could do to mitigate these undesirable
harms?_

    Nothing the authors can think at the time of sharing.

### Are there tasks for which the dataset should not be used?

_If so, please provide a description._

    NO

### Any other comments?

    The data provided is solely for informational and research purposes. Any potential usage of this data is the sole responsibility of the user, and the parties who provided the data will not be held responsible for any actions or decisions made by the user based on this data. It is the responsibility of the user to ensure that any use of the data is legal, ethical, and in compliance with all applicable laws and regulations.

---
## Distribution

### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? 

_If so, please provide a description._

    Yes, the data will be open-sourced at the publication of the related paper. In fact, the goal is to facilitate novel studies in this and related fields by leveraging our annotated data.

### How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?

_Does the dataset have a digital object identifier (DOI)?_

    The dataset will be available for download at: [https://amsacta.unibo.it/id/eprint/7347](https://amsacta.unibo.it/id/eprint/7347)

### When will the dataset be distributed?

    At publication of the related paper.

### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

_If so, please describe this license and/or ToU, and provide a link or other access point to,
or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated
with these restrictions._

<par>
This dataset is shared under <a href="https://creativecommons.org/licenses/by-sa/4.0/">Creative Commons: Attribution-Share Alike 4.0 (CC BY-SA 4.0) License</a>. 

Users must cite:
 - Clissa, L., et al., 2023. [Fluorescent Neuronal Cells v2: Multi-Task, Multi-Format Annotations for Deep Learning in Microscopy.](https://doi.org/<doi-to-be>) Scientific data, (submitted). [Dataset] 

Clissa, L., et al., (2023) [Fluorescent Neuronal Cells v2.](https://amsacta.unibo.it/id/eprint/7347) University of Bologna. [Dataset]  

</par>

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

_If so, please describe these restrictions, and provide a link or other access point to, or
otherwise reproduce, any relevant licensing terms, as well as any fees associated with these
restrictions._

    NO

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?

_If so, please describe these restrictions, and provide a link or other access point to, or otherwise
reproduce, any supporting documentation._

    NO

### Any other comments?

    NONE


---
## Maintenance

_These questions are intended to encourage dataset creators to plan for dataset maintenance
and communicate this plan with dataset consumers._

### Who is supporting/hosting/maintaining the dataset?

    The dataset is thought as a static artifact, and maintenance is not foreseen in the short period. However, maintenance interventions will be performed in case of major errors on a best effort basis.

### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?

<par>
For any inquiry, please refer to <a href="mailto:luca.clissa@bo.infn.it">luca.clissa@bo.infn.it</a>
</par>

### Is there an erratum?

_If so, please provide a link or other access point._

    NO

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?

_If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)?_

    Corrections and integrations are not planned at the moment. However, they may happen in the future, should they become relevant.

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?

_If so, please describe these limits and explain how they will be enforced._

    NOT APPLICABLE

### Will older versions of the dataset continue to be supported/hosted/maintained?

_If so, please describe how. If not, please describe how its obsolescence will be communicated to users._

    Yes, the AMS Acta institutional repository keeps track of artifacts versions.

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

_If so, please provide a description. Will these contributions be validated/verified? If so,
please describe how. If not, why not? Is there a process for communicating/distributing these
contributions to other users? If so, please provide a description._

<par>
This is not foreseen at the moment. However, collaborations are more than welcome in the future. 
For any inquiry, please refer to <a href="mailto:luca.clissa@bo.infn.it">luca.clissa@bo.infn.it</a>
</par>

### Any other comments?

    NONE