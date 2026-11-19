"""Prespecified sparse CT-derived nonfluid hints on authorized positive-only T cases."""
import hashlib
import numpy as np
from scipy import ndimage as nd


RULE = {'grid_mm':2,'body_soft_tissue_threshold_HU':-400,'axial_body_closing_radius_mm':4,
        'inside_body_boundary_mm':10,'enclosed_gas_component_threshold_HU':-650,
        'gas_component_minimum_mL':100,'gas_component_minimum_SI_extent_mm':60,
        'gas_component_selection':'up to largest two satisfying limits and intersecting middle 50% of axial FOV',
        'dense_and_gas_neighborhood':'3 by 3 by 3 cube, including center',
        'gas_maximum_HU':-900,'dense_minimum_HU':700,'distance_from_positive_mm_strictly_greater_than':5,
        'each_group_maximum_points_per_patient':2048,'seed':891,
        'objective':'if any hint inside T training crop, equal weight mean positive BCE and mean weak-negative BCE; otherwise use positive BCE; unknown omitted',
        'scope':'CT-derived sparse intensity cues within T only. Enclosed gas and dense material are heuristic evidence, not clinically verified organ labels or whole-volume background.'}


def body_interior(ct):
    x,y=np.ogrid[-2:3,-2:3]
    circle=x*x+y*y<=4
    out=np.zeros(ct.shape,dtype=bool)
    for k in range(ct.shape[2]):
        soft=nd.binary_closing(ct[:,:,k]>RULE['body_soft_tissue_threshold_HU'],structure=circle)
        labels,n=nd.label(soft,structure=np.ones((3,3),dtype=bool))
        if not n:
            continue
        sizes=np.bincount(labels.ravel()); sizes[0]=0
        body=nd.binary_fill_holes(labels==int(sizes.argmax()))
        distance=nd.distance_transform_edt(np.pad(body,1),sampling=2)[1:-1,1:-1]
        out[:,:,k]=distance>=RULE['inside_body_boundary_mm']
    return out


def lung_like_region(ct, inside):
    components,n=nd.label(inside & (ct<RULE['enclosed_gas_component_threshold_HU']))
    sizes=np.bincount(components.ravel(),minlength=n+1); sizes[0]=0
    eligible=np.flatnonzero(sizes*.008>=RULE['gas_component_minimum_mL'])
    central_ids=set(np.unique(components[:,:,ct.shape[2]//4:3*ct.shape[2]//4]).tolist())
    chosen=[]
    for k in sorted(eligible, key=lambda k:(-sizes[k],k)):
        if k not in central_ids:
            continue
        planes=np.flatnonzero(np.any(components==k, axis=(0,1)))
        if (planes[-1]-planes[0]+1)*2>=RULE['gas_component_minimum_SI_extent_mm']:
            chosen.append(k)
        if len(chosen)==2:
            break
    return np.isin(components,chosen),len(chosen)


def select(ct, known, key):
    inside=body_interior(ct)
    gas_support,n=lung_like_region(ct,inside)
    # Explicit structuring element for Euclidean radius 5 mm on a 2-mm isotropic lattice.
    d=np.arange(-3,4)*2
    ball5=d[:,None,None]**2+d[None,:,None]**2+d[None,None,:]**2<=25
    protected=nd.binary_dilation(known.astype(bool),structure=ball5)
    unit=np.ones((3,3,3),dtype=bool)
    gas=gas_support & nd.binary_erosion(ct<=RULE['gas_maximum_HU'],structure=unit)
    dense=inside & nd.binary_erosion(ct>=RULE['dense_minimum_HU'],structure=unit)
    stats={'gas_components_passing':n,'clinical_background_points':0}
    groups={}
    for name, pool in [('gas',gas),('dense',dense)]:
        rng_seed=int.from_bytes(hashlib.sha256(f"sparse-hu:{RULE['seed']}:{key}:{name}".encode()).digest()[:8],'big')
        eligible=np.flatnonzero(pool & ~protected)
        take=min(RULE['each_group_maximum_points_per_patient'],len(eligible))
        chosen=np.sort(np.random.default_rng(rng_seed).choice(eligible,take,replace=False))
        pointset=np.stack(np.unravel_index(chosen,ct.shape),axis=-1).astype('int32')
        hit=int(known.ravel()[chosen].sum()); near=int(protected.ravel()[chosen].sum())
        if hit or near:
            raise ValueError('Hints intersect target or its exclusion neighborhood')
        groups[name]=pointset
        stats[name]={'before_exclusion':int(pool.sum()),'remaining_candidates':len(eligible),'selected':take,
                     'candidate_at_known_fg':int(np.count_nonzero(pool & known.astype(bool))),
                     'removed_within_5mm':int(np.count_nonzero(pool & protected)),
                     'selected_at_known_positive':hit,'selected_within_5mm':near,
                     'selected_overlap_fraction':0.0 if take else None}
    return groups,stats


def hint_crop(case,coordinates,seed,size):
    ct,_,anchors,_=case
    center=np.asarray(np.unravel_index(np.random.default_rng(seed).choice(anchors),ct.shape))
    start=np.clip(center-size//2,0,np.array(ct.shape)-size)
    mask=np.zeros((size,size,size),dtype='uint8')
    for group,points in enumerate((coordinates['gas'],coordinates['dense']),1):
        local=points-start
        selected=local[np.all((local>=0) & (local<size),axis=1)]
        if len(selected):
            mask[tuple(selected.T)]=group
    return mask
