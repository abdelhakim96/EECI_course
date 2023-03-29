classdef RelationDataPlotter<handle
    properties (Access=private)
        figHMap
        figToFigGrpKeyMap
        figToAxesToHMap
        figToAxesToPlotHMap
        figToAxesKeysMap
        nMaxAxesRows
        nMaxAxesCols
        figureGroupKeySuffFunc
        %
        figureGetNewHandleFunc
        axesGetNewHandleFunc
        axesRearrangeFunc
    end
    properties (Constant,GetAccess=private)
        DEF_N_MAX_AXES_ROWS=2;
        DEF_N_MAX_AXES_COLS=2;
        DEF_FIGURE_GROUP_SUFF_FUNC=@(x)sprintf('_g%d',x);
        DEF_AXES_GET_NEW_HANDLE_FUNC=@(axesKey,...
            nSurfaceRows,nSurfaceColumns,...
            indAxes,hFigureParent,figureKey)...
            subplot(nSurfaceRows,nSurfaceColumns,...
            indAxes,'Parent',hFigureParent);
        DEF_AXES_REARRANGE_FUNC=@(hAxes,axesKey,...
            nSurfaceRows,nSurfaceColumns,...
            indAxes,hFigureParent,figureKey)...
            subplot(nSurfaceRows,nSurfaceColumns,...
            indAxes,hAxes,'Parent',hFigureParent);
        %
        DEF_FIGURE_GET_NEW_HANDLE_FUNC=@(varargin)figure();
    end
    methods
        function SProps=getPlotStructure(self)
            SProps.figHMap=self.figHMap.getCopy();
            SProps.figToFigGrpKeyMap=self.figToFigGrpKeyMap.getCopy();
            SProps.figToAxesToHMap=self.figToAxesToHMap.getCopy();
            SProps.figToAxesToPlotHMap=self.figToAxesToPlotHMap.getCopy();
        end
        function self=RelationDataPlotter(varargin)
            % RELATIONDATAPLOTTER responsible for plotting data from a
            % relation represented by ARelation object
            %
            % Input:
            %   properties:
            %       nMaxAxesRows: numeric[1,1] - maximum number of axes
            %           rows per single figure
            %       nMaxAxesCols: numeric[1,1] - maximum number of axes
            %           columns per single figure
            %       figureGroupKeySuffFunc: function_handle[1,1] - function
            %           responsible for converting a number of figure within a
            %           single group into a character suffix. This function is
            %           useful when a number of axes for some figure is greater
            %           than nMaxAxesRows*nMaxAxesCols in which case the
            %           remainder of the axis is moved to a additional figures
            %           with names composed from figure group name and suffix
            %           produced by figureGroupKeySuffFunc
            %
            %       figureGetNewHandleFunc: function_handle[1,1] -
            %          function, responsible for generating a new
            %          figure handle. The input arguments are
            %
            %               figureGroupKey: char[1,] - group key,
            %                   generated by
            %                   figureGetGroupKeyFunc function passed into
            %                   plotGeneric method
            %               figureGroupKeySuff: char[1,] - group key
            %                   suffix generated by  figureGroupKeySuffFunc
            %
            %               indFigureSubGroup: double[1,1] - number of
            %                   subgroup
            %                   within a group defined by groupKey
            %            Note: if this property is not specified, a simple
            %               hFigure=figure() call is used to generate
            %               a new handle
            %
            %      axesGetNewHandleFunc: function_handle[1,1] - function
            %       responsible for generating a new axes handle based on
            %       the following input arguments:
            %               axesKey: char[1,] - axes key
            %               nSurfaceRows: double[1,1] - number of rows in
            %                   the axis matrix.
            %               nSurfaceColumns: double[1,1] - number of rows
            %                   in the axis matrix
            %               indAxes: double[1,1] - linear index of axis
            %                   within an axes matrix
            %               hFigureParent: double[1,1] - handle of the
            %                   parent figure
            %               figureKey: double[1,1] - figure key
            %
            %           Note: if this property is not specified,
            %               hAxes=subplot(nSurfaceRows,nSurfaceColumns,...
            %                   indAxes,'Parent',hFigure);
            %               call is used to generate a new axis handle.
            %
            %      axesRearrageFunc: function_handle[1,1] - function
            %       responsible for rearranging already existing axes
            %       into a new position (due to newly appreared
            %       axes on the same figure) based on the following input
            %       arguments:
            %               hAxes: double[1,1] - handle of the axes
            %               axesKey: char[1,] - axes key
            %               nSurfaceRows: double[1,1] - number of rows in
            %                   the axis matrix.
            %               nSurfaceColumns: double[1,1] - number of rows
            %                   in the axis matrix
            %               indAxes: double[1,1] - linear index of axis
            %                   within an axes matrix
            %               hFigureParent: double[1,1] - handle of the
            %                   parent figure
            %               figureKey: double[1,1] - figure key
            %
            %           Note: if this property is not specified,
            %               subplot(nSurfaceRows,nSurfaceColumns,...
            %                   indAxes,hAxes,'Parent',hFigure);
            %               call is used to rearrange axes.
            %
            % $Author: Peter Gagarinov  <pgagarinov@gmail.com> $	$Date: 2012-01-12 $
            % $Copyright: Moscow State University,
            %            Faculty of Computational Mathematics and Computer Science,
            %            System Analysis Department 2012 $
            %
            self.figHMap=modgen.containers.MapExtended();
            self.figToFigGrpKeyMap=modgen.containers.MapExtended();
            self.figToAxesToHMap=modgen.containers.MapExtended();
            self.figToAxesToPlotHMap=modgen.containers.MapExtended();
            self.figToAxesKeysMap=modgen.containers.MapExtended();
            %
            [~,~,self.nMaxAxesRows,self.nMaxAxesCols,...
                self.figureGroupKeySuffFunc,self.figureGetNewHandleFunc,...
                self.axesGetNewHandleFunc,self.axesRearrangeFunc]=...
                modgen.common.parseparext(varargin,...
                {'nMaxAxesRows','nMaxAxesCols','figureGroupKeySuffFunc',...
                'figureGetNewHandleFunc','axesGetNewHandleFunc',...
                'axesRearrangeFunc';...
                self.DEF_N_MAX_AXES_ROWS,self.DEF_N_MAX_AXES_COLS,...
                self.DEF_FIGURE_GROUP_SUFF_FUNC,...
                self.DEF_FIGURE_GET_NEW_HANDLE_FUNC,...
                self.DEF_AXES_GET_NEW_HANDLE_FUNC,...
                self.DEF_AXES_REARRANGE_FUNC;...
                'isnumeric(x)&&isscalar(x)','isnumeric(x)&&isscalar(x)',...
                'isfunction(x)&&isscalar(x)',...
                'isfunction(x)&&isscalar(x)',...
                'isfunction(x)&&isscalar(x)',...
                'isfunction(x)&&isscalar(x)'},0);
        end
        %%
        function closeAllFigures(self)
            % CLOSEALLFIGURES closes all figures
            import modgen.logging.log4j.Log4jConfigurator;
            logger=Log4jConfigurator.getLogger();
            %
            mp=self.figHMap;
            cellfun(@closeFigure,mp.values);
            self.clearGraphicHandleMaps();
            function closeFigure(h)
                if ishandle(h)
                    close(h);
                else
                    logger.warn([num2str(h),' is invalid figure handle']);
                end
            end
        end
        %%
        function saveAllFigures(self,resFolderName,formatNameList)
            % SAVEALLFIGURES saves all figures to a specified folder in
            % 'fig' format
            %
            % Input:
            %   regular:
            %       self:
            %       resFolderName: char[1,] - destination folder name
            %
            %   optional:
            %       formatNameList: char[1,]/cell[1,] of char[1,]
            %           - list of format names accepted by the built-in
            %           "saveas" function, default value is 'fig';
            %
            %
            mp=self.figHMap;
            hFigureList=mp.values;
            if isempty(hFigureList)
                hFigureVec=gobjects(1,0);
            else
                hFigureVec=[hFigureList{:}];
            end
            %
            fileNameList=cellfun(@modgen.common.genfilename,...
                mp.keys,'UniformOutput',false);
            %
            if nargin<3
                formatNameList={'fig'};
            end
            %
            modgen.graphics.savefigures(hFigureVec,resFolderName,...
                formatNameList,fileNameList)
        end
        %%
        function plotGeneric(self,rel,...
                figureGetGroupKeyFunc,figureGetGroupKeyFieldNameList,...
                figureSetPropFunc,figureSetPropFieldNameList,...
                axesGetKeyFunc,axesGetKeyFieldNameList,...
                axesSetPropFunc,axesSetPropFieldNameList,...
                plotCreateFunc,plotCreateFieldNameList,varargin)
            % PLOTGENERIC plots a content of specified ARelation object
            %
            % Input:
            %   regular:
            %       self:
            %       rel: smartdb.relation.ARelation[1,1] - relation
            %           containing the data to plot
            %
            %       figureGetGroupKeyFunc: function_handle[1,1]
            %                       /cell[1,nFuncs] of function_handle[1,1]
            %           - function responsible for producing figure group
            %             name.
            %       figureGetGroupKeyFieldNameList: cell[1,] of char[1,] -
            %           list of fields of specified relation (rel) passed
            %           into figureGetGroupKeyFunc as input arguments
            %
            %       figureSetPropFunc: function_handle[1,1]/
            %                   cell[1,nFuncs] of function_handle[1,1]
            %           - function(s) responsible for setting properties of
            %           figure objects, the first argument to the
            %           function is a handle of the corresponding figure,
            %           the second one is figureKey, the third one is
            %           figure group number while the rest are defined by
            %           the following property
            %       figureSetPropFieldNameList: cell[1,] of char[1,] - list of
            %           fields of specified relations passed into
            %           figureSetPropFunc as additional input arguments
            %
            %       axesGetKeyFunc: function_handle[1,1]/
            %                   cell[1,nFuncs] of function_handle[1,1]
            %           - handle of function(s) responsible for
            %           generating an axes name
            %       axesGetKeyFieldNameList: cell[1,] of char[1,] - list of fields
            %           of the specified relation passed into
            %           axesGetKeyFunc as input arguments
            %
            %       axesSetPropFunc: function_handle[1,1]/
            %                   cell[1,nFuncs] of function_handle[1,1]
            %           - handle of function(s) responsible for
            %           setting axes properties  the first
            %           argument is axes handle, the second one is
            %           axes key while the rest of the arguments defined by
            %           axesSetPropFieldNameList property
            %
            %       axesSetPropFieldNameList: cell[1,] of char[1,] - list of fields
            %           of the specified relation passed into
            %           axesSetPropFunc as input arguments,             %
            %
            %       plotCreateFunc: function_handle[1,1]/
            %               cell[1,nFuncs] of function_handle[1,1]
            %           - function(s) responsible for plotting data
            %           from the specified relation on
            %           the axes specified by handle passed as
            %           the first input argument to the function
            %       plotCreateFieldNameList: cell[1,] of char[1,] - list of fields
            %           of the specified relation passed into
            %           plotCreateFunc as additional input arguments
            %
            %   properties:
            %       axesPostPlotFunc: function_handle[1,1]/
            %                   cell[1,nFuncs] of function_handle[1,1]
            %           - handle of function(s) responsible for
            %           setting axes properties after the drawing process is finished
            %           the first argument is axes handle, the
            %           second one is axes key while the rest of them are
            %           defined by axesSetPropFieldNameList
            %       isAutoHoldOn: logical[1,1] - if true, hold(hAxes,'on')
            %           is called automatically for every axes
            %
            %
            import modgen.logging.log4j.Log4jConfigurator;
            import modgen.common.type.simple.*;
            import smartdb.disp.RelationDataPlotter;
            import modgen.common.throwerror;
            import modgen.struct.updateleaves;
            import modgen.common.type.simple.checkcelloffunc;
            import modgen.common.throwwarn;
            %
            [~,~,axesPostPlotFunc,isAutoHoldOn,isAxesPostPlotFuncSpec]=...
                modgen.common.parseparext(...
                varargin,...
                {'axesPostPlotFunc','isAutoHoldOn';...
                [],true;...
                'true','isscalar(x)&&islogical(x)'},...
                0);
            logger=Log4jConfigurator.getLogger();
            %
            checkcellofstr(figureGetGroupKeyFieldNameList,true);
            checkcellofstr(axesGetKeyFieldNameList,true);
            %
            checkcellofstr(figureSetPropFieldNameList,true);
            checkcellofstr(axesSetPropFieldNameList,true);
            checkcellofstr(plotCreateFieldNameList,true);
            %
            [figureGetGroupKeyFunc,figureSetPropFunc,...
                axesGetKeyFunc,axesSetPropFunc,plotCreateFunc]=...
                RelationDataPlotter.checkFunc(...
                figureGetGroupKeyFunc,figureSetPropFunc,...
                axesGetKeyFunc,axesSetPropFunc,plotCreateFunc);
            %
            if isAxesPostPlotFuncSpec
                [~,axesPostPlotFunc]=...
                    RelationDataPlotter.checkFunc(...
                    axesSetPropFunc,axesPostPlotFunc);
            end
            %
            nFuncs=length(figureGetGroupKeyFunc);
            nEntries=rel.getNTuples();
            %
            figureGroupKeyCMat=RelationDataPlotter.cellFunArray(...
                rel,figureGetGroupKeyFunc,...
                figureGetGroupKeyFieldNameList);
            axesKeyCMat=RelationDataPlotter.cellFunArray(rel,...
                axesGetKeyFunc,axesGetKeyFieldNameList);
            %
            figureGroupKeyList=figureGroupKeyCMat(:);
            [uniqueFigureGroupKeyList,~,indUniqueFigureGroupVec]=...
                unique(figureGroupKeyList);
            prevAxesKeyList=cellfun(@self.getAxesKeyList,...
                uniqueFigureGroupKeyList,'UniformOutput',false);
            SData.figureGroupKey=figureGroupKeyList;
            SData.axesKey=axesKeyCMat(:);
            figAxesMapRel=smartdb.relations.DynamicRelation(SData);
            figAxesMapRel.removeDuplicateTuples();
            figAxesMapRel.groupBy('figureGroupKey');
            figAxesMapRel=figAxesMapRel.getTuplesIndexedBy(...
                'figureGroupKey',figureGroupKeyList);
            %
            figurePropCMat=rel.toMat('fieldNameList',figureSetPropFieldNameList);
            axesPropCMat=rel.toMat('fieldNameList',axesSetPropFieldNameList);
            plotPropCMat=rel.toMat('fieldNameList',plotCreateFieldNameList);
            %
            prevAxesUnqCMat=reshape(...
                prevAxesKeyList(indUniqueFigureGroupVec),nEntries,nFuncs);
            axesUnqCMat=reshape(figAxesMapRel.axesKey,nEntries,nFuncs);
            isPrevAxesMat=~cellfun('isempty',prevAxesUnqCMat);
            if any(isPrevAxesMat(:))
                axesUnqCMat(isPrevAxesMat)=cellfun(...
                    @(x,y)[x(:);reshape(y(~ismember(y,x)),[],1)],...
                    prevAxesUnqCMat(isPrevAxesMat),...
                    axesUnqCMat(isPrevAxesMat),'UniformOutput',false);
            end
            %
            if nEntries>0
                %
                figureKeyCMat=cell(nEntries,nFuncs);
                hAxesMat=zeros(nEntries,nFuncs);
                hPlotCMat=cell(nEntries,nFuncs);
                for iEntry=1:nEntries
                    for iFunc=1:nFuncs
                        figureGroupKey=figureGroupKeyCMat{iEntry,iFunc};
                        axesKey=axesKeyCMat{iEntry,iFunc};
                        %
                        [hAxesMat(iEntry,iFunc),...
                            figureKeyCMat{iEntry,iFunc}]=...
                            self.getAxesHandle(figureGroupKey,...
                            figureSetPropFunc{iFunc},figurePropCMat(iEntry,:),...
                            axesUnqCMat{iEntry,iFunc},axesKey);
                        %
                    end
                end
                % fill names of axes for each figure preserving their order
                % in axesUnqCMat
                [uniqueFigureKeyList,indForwardVec,indBackwardVec]=...
                    unique(figureKeyCMat(:));
                nUniqueFigureKeys=numel(uniqueFigureKeyList);
                for iUniqueFigureKey=1:nUniqueFigureKeys
                    figureKey=uniqueFigureKeyList{iUniqueFigureKey};
                    isFigureKeyVec=indBackwardVec==iUniqueFigureKey;
                    axesKeyList=unique(reshape(...
                        axesKeyCMat(isFigureKeyVec),[],1));
                    curInd=indForwardVec(iUniqueFigureKey);
                    allAxesKeyList=reshape(axesUnqCMat{curInd},[],1);
                    if isPrevAxesMat(curInd)
                        allAxesKeyList(ismember(allAxesKeyList,...
                            prevAxesUnqCMat{curInd}))=[];
                        if isempty(allAxesKeyList)
                            continue;
                        end
                        if self.figToAxesKeysMap.isKey(figureKey)
                            prevAxesKeyList=reshape(...
                                self.figToAxesKeysMap(figureKey),[],1);
                        else
                            prevAxesKeyList={};
                        end
                    else
                        prevAxesKeyList={};
                    end
                    self.figToAxesKeysMap(figureKey)=vertcat(...
                        prevAxesKeyList,allAxesKeyList(ismember(...
                        allAxesKeyList,axesKeyList)));
                end
                if ~isequal(self.figToAxesToHMap.keys,...
                        self.figToAxesKeysMap.keys)
                    modgen.common.throwerror('badFigToAxesKeysMap',[...
                        'figToAxesKeysMap is not consistent with '...
                        'figToAxesToHMap']);
                end
                if self.figToAxesToHMap.Count > 0
                    allFigureKeyList = self.figToAxesToHMap.keys;
                    for iFigureKey = 1:numel(allFigureKeyList)
                        figureKey=allFigureKeyList{iFigureKey};
                        if ~isequal(reshape(...
                                self.figToAxesToHMap(figureKey).keys,...
                                [],1),reshape(sort(...
                                self.figToAxesKeysMap(figureKey)),[],1))
                            modgen.common.throwerror(...
                                'badFigToAxesKeysMap',['Values of '...
                                'figToAxesKeysMap is not consistent with '...
                                'figToAxesToHMap for figureKey=%s'],...
                                figureKey);
                        end
                    end
                end
                %% fill figToAxesToPlotHMap with the plot handles
                %  not registered in the RelationDataPlotter
                %
                figureKeyUList=self.figToAxesToHMap.keys;
                nUniqueFigureKeys=length(figureKeyUList);
                for iFigureKey=1:nUniqueFigureKeys
                    figureKey=figureKeyUList{iFigureKey};
                    %
                    axesMap=self.figToAxesToHMap(figureKey);
                    axisUKeyList=axesMap.keys;
                    nUAxis=length(axisUKeyList);
                    if self.figToAxesToPlotHMap.isKey(figureKey);
                        axisToPlotMap=self.figToAxesToPlotHMap(figureKey);
                    else
                        axisToPlotMap=modgen.containers.MapExtended();
                        self.figToAxesToPlotHMap(figureKey)=axisToPlotMap;
                    end
                    for iAxes=1:nUAxis
                        axesKey=axisUKeyList{iAxes};
                        hAxes=axesMap(axesKey);
                        axisToPlotMap(axesKey)=findall(hAxes,'Parent',...
                            hAxes).';
                    end
                end
                
                %
                for iEntry=1:nEntries
                    for iFunc=1:nFuncs
                        hPlotCMat{iEntry,iFunc}=setAxesProp(self,...
                            figureKeyCMat{iEntry,iFunc},...
                            axesKeyCMat{iEntry,iFunc},....
                            axesSetPropFunc{iFunc},...
                            axesPropCMat(iEntry,:),isAutoHoldOn);
                    end
                end
                %
                for iEntry=1:nEntries
                    for iFunc=1:nFuncs
                        hPlotCMat{iEntry,iFunc}=[hPlotCMat{iEntry,iFunc},...
                            plotCreateFunc{iFunc}(...
                            hAxesMat(iEntry,iFunc),...
                            plotPropCMat{iEntry,:})];
                    end
                end
                if isAxesPostPlotFuncSpec
                    for iEntry=1:nEntries
                        for iFunc=1:nFuncs
                            hPlotCMat{iEntry,iFunc}=...
                                [hPlotCMat{iEntry,iFunc},setAxesProp(self,...
                                figureKeyCMat{iEntry,iFunc},...
                                axesKeyCMat{iEntry,iFunc},....
                                axesPostPlotFunc{iFunc},...
                                axesPropCMat(iEntry,:),isAutoHoldOn)];
                        end
                    end
                end
                %% Complement figToAxesToPlotHMap with the plot
                % handles of newly created graphs
                %
                [figureKeyUList,~,indUVec]=unique(figureKeyCMat(:));
                nUniqueFigureKeys=length(figureKeyUList);
                for iFigureKey=1:nUniqueFigureKeys
                    figureKey=figureKeyUList{iFigureKey};
                    %
                    isCurAxisVec=indUVec==iFigureKey;
                    axisKeyList=axesKeyCMat(isCurAxisVec);
                    plotHVecList=hPlotCMat(isCurAxisVec);
                    [axisUKeyList,~,indUAVec]=unique(axisKeyList);
                    nUAxis=length(axisUKeyList);
                    plotHandleUKeyList=cell(1,nUAxis);
                    axisToPlotMap=self.figToAxesToPlotHMap(figureKey);
                    for iAxes=1:nUAxis
                        plotHandleUKeyList{iAxes}=[plotHVecList{indUAVec==iAxes}];
                        axisKey=axisUKeyList{iAxes};
                        existHandleVec=axisToPlotMap(axisKey);
                        newHandleVec=plotHandleUKeyList{iAxes};
                        catHandleVec=[existHandleVec,newHandleVec];
                        [uniqueHandleVec,indUniqueVec]=unique(catHandleVec);
                        nCatHandles=numel(catHandleVec);
                        if nCatHandles>numel(indUniqueVec)
                            indNotThereVec=setdiff(1:nCatHandles,indUniqueVec);
                            for indNotThere=indNotThereVec
                                notThereHandle=catHandleVec(indNotThere);
                                if ~(ishghandle(notThereHandle)&&...
                                        strcmp(get(notThereHandle,'Type'),'text'))
                                    parentHandle=notThereHandle.Parent;
                                    throwwarn('wrongInput',...
                                        ['an existing graphic object %s:%s',...
                                        ' with parent %s:%s has been ',...
                                        'returned by one of plotting ',...
                                        'functions'],evalc('disp(notThereHandle)'),...
                                        class(notThereHandle),...
                                        evalc('disp(parentHandle)'),...
                                        class(parentHandle));
                                end
                            end
                        end
                        %
                        axisToPlotMap(axisKey)=uniqueHandleVec;
                    end
                end
                %% Check that all plotting handlers were returned by
                % plotCreate function
                SFigToAxes=updateleaves(...
                    self.figToAxesToHMap.toStruct(),...
                    @(x,y)sort(findall(x,'Parent',x).'));
                SExpFigToAxes=updateleaves(...
                    self.figToAxesToPlotHMap.toStruct(),@(x,y)sort(x));
                [isOk,reportStr]=modgen.struct.structcompare(...
                    SExpFigToAxes,SFigToAxes);
                if ~isOk
                    throwerror('wrongInput',...
                        ['Not all figure handlers were registered, ',...
                        'reason %s'],reportStr);
                end
                %
                logger.debug('Storing graphs: done');
            else
                logger.debug('There is nothing to plot');
            end
        end
    end
    %
    methods (Static,Access=private)
        function checkIfValueUnique(mapObj,errorTag,errorMsg)
            valueList=mapObj.values;
            isPos=modgen.common.isunique([valueList{:}]);
            if ~isPos
                modgen.common.throwerror(errorTag,errorMsg);
            end
            
        end
        function resCMat=cellFunArray(rel,fHandleList,fieldNameList)
            nTuples=rel.getNTuples();
            nFuncs=length(fHandleList);
            nFields=length(fieldNameList);
            resCMat=cell(nTuples,nFuncs);
            %
            if nFields>0
                figArgList=rel.toArray('fieldNameList',fieldNameList,...
                    'groupByColumns',true);
                for iFunc=1:nFuncs
                    resCMat(:,iFunc)=cellfun(fHandleList{iFunc},figArgList{:},...
                        'UniformOutput',false);
                end
            else
                for iFunc=1:nFuncs
                    resCMat(:,iFunc)=repmat({fHandleList{iFunc}()},nTuples,1);
                end
            end
        end
        function varargout=checkFunc(varargin)
            import modgen.common.type.simple.checkcelloffunc;
            import modgen.common.throwerror;
            nElemVec=zeros(1,nargin);
            varargout=cell(1,nargin);
            for iArg=1:nargin
                varargout{iArg}=checkcelloffunc(varargin{iArg});
                nElemVec(iArg)=numel(varargout{iArg});
            end
            nFuncs=max(nElemVec);
            isScalarVec=nElemVec==1;
            if ~all(isScalarVec|nElemVec==nFuncs);
                throwerror('wrongInput',...
                    'size of function arrrays should be the same');
            end
            varargout(isScalarVec)=cellfun(@(x)repmat(x,1,nFuncs),...
                varargout(isScalarVec),'UniformOutput',false);
        end
    end
    methods (Access=private)
        function hFigure=getFigureHandle(self,figureGroupKey,...
                figureGroupKeySuff,...
                figureSetPropFunc,figurePropValList,indFigureSubGroup)
            persistent logger;
            import modgen.logging.log4j.Log4jConfigurator;
            import modgen.common.checkvar;
            if isempty(logger)
                logger=Log4jConfigurator.getLogger();
            end
            mp=self.figHMap;
            %
            figureKey=[figureGroupKey,figureGroupKeySuff];
            if mp.isKey(figureKey)
                hFigure=mp(figureKey);
            else
                hFigure=self.figureGetNewHandleFunc(figureGroupKey,...
                    figureGroupKeySuff,indFigureSubGroup);
                %
                checkvar(hFigure,@(x)isscalar(x)&&ishandle(x)&&...
                    strcmp(get(x,'Type'),'figure'),...
                    'errorTag','badFigureHandle',...
                    ['a result of figureGetNewHandleFunc function',...
                    'is expected to be a figure handle']);
                %
                figureSetPropFunc(hFigure,figureKey,indFigureSubGroup,...
                    figurePropValList{:});
                logger.debug(['Figure ',figureKey,...
                    ' is created, hFigure=',num2str(double(hFigure))]);
                mp(figureKey)=hFigure;
                self.figToFigGrpKeyMap(figureKey)=figureGroupKey;
                
                self.checkIfValueUnique(mp,'wrongFigureHandle',...
                    ['handles returned by figureGetNewHandleFunc ',...
                    'should be different for different values ',...
                    'of figureKey']);
                if ~isequal(mp.keys,self.figToFigGrpKeyMap.keys)
                    modgen.common.throwerror('badFigToFigGrpKeyMap',[...
                        'figToFigGrpKeyMap is not consistent with '...
                        'figHMap']);
                end
               
            end
        end
        %%
        function [hAxes,figureKey]=getAxesHandle(self,figureGroupKey,...
                figureSetPropFunc,figurePropValList,...
                axesKeyList,axesKey)
            import modgen.logging.log4j.Log4jConfigurator;
            import modgen.common.checkvar;
            import modgen.common.throwerror;
            persistent logger;
            if isempty(logger)
                logger=Log4jConfigurator.getLogger();
            end
            %
            nMaxAxes=self.nMaxAxesCols*self.nMaxAxesRows;
            %
            nTotalAxis=length(axesKeyList);
            indTotalMetric=find(strcmp(axesKey,axesKeyList));
            indFigureSubGroup=ceil(indTotalMetric/nMaxAxes);
            figureGroupKeySuff=self.figureGroupKeySuffFunc(...
                indFigureSubGroup);
            %
            figureKey=[figureGroupKey,figureGroupKeySuff];
            %
            nFullFigures=fix(nTotalAxis/nMaxAxes);
            if indTotalMetric>nFullFigures*nMaxAxes
                nCurMetrics=rem(nTotalAxis-1,nMaxAxes)+1;
            else
                nCurMetrics=nMaxAxes;
            end
            %
            indAxes=rem(indTotalMetric-1,nMaxAxes)+1;
            %
            mp=self.getAxesHandleMap(figureKey);
            %
            if mp.isKey(axesKey)
                hAxes=mp(axesKey);
            else
                hFigure=self.getFigureHandle(figureGroupKey,...
                    figureGroupKeySuff,...
                    figureSetPropFunc,figurePropValList,indFigureSubGroup);
                %
                nSurfaceRows=min(self.nMaxAxesRows,nCurMetrics);
                nSurfaceColumns=ceil(nCurMetrics/nSurfaceRows);
                %
                logger.debug(['creating axes for figure: ',figureKey,...
                    ',size: ',mat2str([nSurfaceRows,nSurfaceColumns]),...
                    ',position: ',num2str(indAxes)]);
                %
                fanm=self.figToAxesKeysMap;
                if fanm.isKey(figureKey)
                    prevAxesKeyList=fanm(figureKey);
                    nPrevAxes=numel(prevAxesKeyList);
                    if numel(prevAxesKeyList)>=indAxes
                        throwerror('badFigToAxesKeysMap',[...
                            'number of axes registered in '...
                            'figToAxesKeysMap is not consistent with '...
                            'index of newly added axes']);
                    end
                    for iPrevAxes=1:nPrevAxes
                        prevAxesKey=prevAxesKeyList{iPrevAxes};
                        if ~mp.isKey(prevAxesKey)
                            throwerror('badFigToAxesKeysMap',[...
                                'can not find handles for some axes '...
                                'registered in figToAxesKeysMap for '...
                                'given figure']);
                        end
                        hPrevAxes=mp(prevAxesKey);
                        self.axesRearrangeFunc(hPrevAxes,prevAxesKey,...
                            nSurfaceRows,nSurfaceColumns,iPrevAxes,...
                            hFigure,figureKey);
                    end
                end
                hAxes=self.axesGetNewHandleFunc(axesKey,nSurfaceRows,...
                    nSurfaceColumns,indAxes,hFigure,figureKey);
                checkvar(hAxes,@(x)isscalar(x)&&ishandle(x)&&...
                    strcmp(get(x,'Type'),'axes')&&...
                    get(x,'Parent')==hFigure,'errorTag','badAxesHandle',...
                    ['a result of axesGetNewHandleFunc function',...
                    'is expected to be an axes handle with ',...
                    'both a proper parent ',...
                    'and a correct type']);
                
                %
                mp(axesKey)=hAxes;
                self.checkIfValueUnique(mp,'wrongAxesHandle',...
                    ['handles returned by axesGetNewHandleFunc ',...
                    'should be different for different values ',...
                    'of indAxes']);
                
            end
        end
        %%
        function hPlotVec=setAxesProp(self,figureKey,axesKey,...
                axesSetPropFunc,axesPropValList,isAutoHoldOn)
            
            mp=self.getAxesHandleMap(figureKey);
            hAxes=mp(axesKey);
            hPlotVec=feval(axesSetPropFunc,hAxes,axesKey,...
                axesPropValList{:});
            if isAutoHoldOn
                hold(hAxes,'on');
            end
        end
        %%
        function axesKeyList=getAxesKeyList(self,figureGroupKey)
            fnm=self.figToFigGrpKeyMap;
            isFigureGroupVec=strcmp(fnm.values,figureGroupKey);
            if any(isFigureGroupVec)
                figureKeyList=fnm.keys;
                figureKeyList(~isFigureGroupVec)=[];
            else
                axesKeyList={};
                return;
            end
            nFigureKeys=numel(figureKeyList);
            axesKeyListCVec=cell(nFigureKeys);
            fanm=self.figToAxesKeysMap;
            for iFigureKey=1:nFigureKeys
                figureKey=figureKeyList{iFigureKey};
                if fanm.isKey(figureKey)
                    axesKeyListCVec{iFigureKey}=fanm(figureKey);
                end
            end
            axesKeyList=vertcat(axesKeyListCVec{:});
        end
        %%
        function res=getAxesHandleMap(self,figureKey)
            fm=self.figToAxesToHMap;
            if ~fm.isKey(figureKey)
                am=modgen.containers.MapExtended();
                fm(figureKey)=am; %#ok<NASGU>
            else
                am=fm(figureKey);
            end
            res=am;
        end
        %%

        %%
        function clearGraphicHandleMaps(self)
            mp=self.figHMap;
            mp.remove(mp.keys);
            %
            fnm=self.figToFigGrpKeyMap;
            fnm.remove(fnm.keys);
            %
            fm=self.figToAxesToHMap;
            fm.remove(fm.keys);
            %
            pm=self.figToAxesToPlotHMap;
            pm.remove(pm.keys);
            %
            fanm=self.figToAxesKeysMap;
            fanm.remove(fanm.keys);
        end
    end
end
%%